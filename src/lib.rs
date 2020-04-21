use nom::bytes::complete::{tag, take, take_while1};
use nom::character::complete::{line_ending, multispace0, multispace1};
use nom::combinator::{opt, rest};
use nom::multi::many1;
use nom::number::complete::{float, le_f32};
use nom::{eat_separator, named, IResult};
use std::{
    convert::TryInto,
    io::{Read, Seek, SeekFrom},
};

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    IOError(std::io::Error),
    NomError(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            e => Some(e),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::IOError(error)
    }
}

impl<E: std::fmt::Debug> From<nom::Err<E>> for Error {
    fn from(error: nom::Err<E>) -> Self {
        Error::NomError(format!("{}", error))
    }
}

pub trait XYZ: Clone + Copy + From<[f32; 3]> {}

impl XYZ for [f32; 3] {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triangle<N, V> {
    normal: N,
    vertices: [V; 3],
}

unsafe impl<N, V> Send for Triangle<N, V> {}
unsafe impl<N, V> Sync for Triangle<N, V> {}

impl<N: XYZ, V: XYZ> Triangle<N, V> {
    pub fn new(normal: N, vertices: [V; 3]) -> Self {
        Triangle { normal, vertices }
    }

    pub fn normal(&self) -> N {
        self.normal
    }

    pub fn vertices(&self) -> [V; 3] {
        self.vertices
    }

    pub fn size_of(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mesh<N: XYZ, V: XYZ> {
    triangles: Vec<Triangle<N, V>>,
}

impl<N: XYZ, V: XYZ> Mesh<N, V> {
    pub fn new(triangles: Vec<Triangle<N, V>>) -> Self {
        Self { triangles }
    }

    pub fn triangles(&self) -> &[Triangle<N, V>] {
        self.triangles.as_slice()
    }

    pub fn size_of(&self) -> usize {
        let struct_size = std::mem::size_of::<Self>();
        let triangles_size = self.triangles.len() * std::mem::size_of::<Triangle<N, V>>();
        struct_size + triangles_size
    }
}

// BOTH GRAMMARS
/////////////////////////////////////////////////////////////////

/// Parse a binary or an ASCII stl.
/// Binary stls ar not supposed to begin with the bytes `solid`,
/// but unfortunately they sometimes do in the real world.
/// For this reason, we use a simple regex heuristic to determine
/// if the stl contains the bytes `facet normal`, which is a byte
/// sequence specifically used in ASCII stls.
/// If the file contains this sequence, we assume ASCII, otherwise
/// binary. While a binary stl can in theory contain this sequence,
/// the odds of this are low. This is a tradeoff to avoid something
/// both more complicated and less performant.
pub fn parse_stl<R: Read + Seek, N: XYZ, V: XYZ>(bytes: &mut R) -> Result<Mesh<N, V>> {
    if contains_facet_normal_bytes(bytes.by_ref()) {
        bytes.seek(SeekFrom::Start(0))?;

        let mut buf = vec![];

        bytes.read_to_end(&mut buf)?;

        mesh_ascii(&buf)
    } else {
        bytes.seek(SeekFrom::Start(0))?;
        mesh_binary(bytes.by_ref())
    }
}

fn contains_facet_normal_bytes<R: Read>(bytes: &mut R) -> bool {
    let identifier_search_bytes_length = match std::env::var("NOM_IDENTIFIER_SEARCH_BYTES_LENGTH") {
        Ok(length) => length.parse().unwrap_or_else(|_| 1024),
        Err(_e) => 1024,
    };

    let mut search_space = vec![0u8; identifier_search_bytes_length];

    bytes.read_to_end(&mut search_space).unwrap();

    search_bytes(&search_space, b"facet normal").is_some()
}

fn search_bytes(bytes: &[u8], target: &[u8]) -> Option<usize> {
    bytes
        .windows(target.len())
        .position(|window| window == target)
}

// BINARY GRAMMAR
/////////////////////////////////////////////////////////////////
//
// Format of a binary STL:
//
// UINT8[80] – Header
// UINT32 – Number of triangles
//
// foreach triangle
// REAL32[3] – Normal vector
// REAL32[3] – Vertex 1
// REAL32[3] – Vertex 2
// REAL32[3] – Vertex 3
// UINT16 – Attribute byte count
// end
//
// Therefor we see that the size of an STL is:
// 80 byte header
// + 4 byte triangle size
// + (n * (12 + 12 + 12 + 12 + 2))

const HEADER_SIZE_BYTES: usize = 84; // 80 + 4
const TRIANGLE_SIZE_BYTES: usize = 50; // 12 + 12 + 12 + 12 + 2

fn mesh_binary<R: Read, N: XYZ, V: XYZ>(mut s: R) -> Result<Mesh<N, V>> {
    let mut header_and_triangles_count = [0u8; HEADER_SIZE_BYTES];

    s.read_exact(&mut header_and_triangles_count)?;

    let reported_triangle_count = u32::from_le_bytes(
        header_and_triangles_count[80..84]
            .try_into()
            .expect("Could not get four bytes to create u32"),
    );

    let mut all_triangles: Vec<Triangle<N, V>> =
        Vec::with_capacity(reported_triangle_count as usize);

    let triangles_reader: TrianglesIter<R, N, V> =
        TrianglesIter::new(s, reported_triangle_count as usize);

    for triangle in triangles_reader {
        all_triangles.push(triangle?);
    }

    let mesh = Mesh::new(all_triangles);

    Ok(mesh)
}
#[derive(Debug)]
pub struct TrianglesIter<R: Read, N: XYZ, V: XYZ> {
    reader: R,
    buf: Vec<u8>,
    triangles_to_read: usize,
    triangles_read: usize,
    normal_type: std::marker::PhantomData<N>,
    vertex_type: std::marker::PhantomData<V>,
}

impl<R: Read, N: XYZ, V: XYZ> TrianglesIter<R, N, V> {
    fn new(reader: R, triangles_to_read: usize) -> Self {
        TrianglesIter {
            reader,
            buf: vec![0u8; TRIANGLE_SIZE_BYTES],
            triangles_to_read,
            triangles_read: 0,
            normal_type: std::marker::PhantomData,
            vertex_type: std::marker::PhantomData,
        }
    }
}

impl<R: Read, N: XYZ, V: XYZ> Iterator for TrianglesIter<R, N, V> {
    type Item = Result<Triangle<N, V>>;

    fn next(&mut self) -> Option<Result<Triangle<N, V>>> {
        if self.triangles_read >= self.triangles_to_read {
            None
        } else {
            match self.reader.read_exact(&mut self.buf) {
                Ok(()) => match triangle_binary(&self.buf) {
                    Ok((_r, t)) => {
                        self.triangles_read += 1;
                        Some(Ok(t))
                    }
                    Err(err) => {
                        self.triangles_read += 1;
                        Some(Err(Error::from(err)))
                    }
                },
                Err(e) => Some(Err(Error::from(e))),
            }
        }
    }
}

fn three_f32s(s: &[u8]) -> IResult<&[u8], [f32; 3]> {
    let (s, f1) = le_f32(s)?;
    let (s, f2) = le_f32(s)?;
    let (s, f3) = le_f32(s)?;

    Ok((s, [f1, f2, f3]))
}

fn triangle_binary<N: XYZ, V: XYZ>(s: &[u8]) -> IResult<&[u8], Triangle<N, V>> {
    let (s, normal) = three_f32s(s)?;
    let (s, v1) = three_f32s(s)?;
    let (s, v2) = three_f32s(s)?;
    let (s, v3) = three_f32s(s)?;
    let (s, _attribute_byte_count) = take(2usize)(s)?;

    Ok((
        s,
        Triangle {
            normal: N::from(normal),
            vertices: [V::from(v1), V::from(v2), V::from(v3)],
        },
    ))
}

// ASCII GRAMMAR
/////////////////////////////////////////////////////////////////

fn not_line_ending(c: u8) -> bool {
    c != b'\r' && c != b'\n'
}

fn mesh_ascii<'a, N: XYZ, V: XYZ>(s: &'a [u8]) -> Result<Mesh<N, V>> {
    let (s, _) = tag("solid ")(s).map_err(to_crate_err)?;
    let (s, _) = opt(take_while1(not_line_ending))(s).map_err(to_crate_err)?;
    let (s, _) = line_ending(s).map_err(to_crate_err)?;
    let (s, triangles) = many1(triangle_ascii)(s)?;
    let (s, _) = multispace1(s).map_err(to_crate_err)?;
    let (s, _) = tag("endsolid")(s).map_err(to_crate_err)?;
    let (_s, _) = opt(rest)(s).map_err(to_crate_err)?;
    let mesh = Mesh::new(triangles);

    Ok(mesh)
}

#[inline(always)]
fn to_crate_err(e: nom::Err<(&[u8], nom::error::ErrorKind)>) -> Error {
    e.into()
}

named!(whitespace, eat_separator!(&b" \t\r\n"[..]));

fn three_floats(s: &[u8]) -> IResult<&[u8], [f32; 3]> {
    let (s, f1) = float(s)?;
    let (s, _) = multispace1(s)?;
    let (s, f2) = float(s)?;
    let (s, _) = multispace1(s)?;
    let (s, f3) = float(s)?;

    Ok((s, [f1, f2, f3]))
}

fn triangle_ascii<N: XYZ, V: XYZ>(s: &[u8]) -> IResult<&[u8], Triangle<N, V>> {
    let (s, _) = multispace0(s)?;
    let (s, _) = tag("facet normal")(s)?;
    let (s, _) = multispace1(s)?;

    let (s, normal) = three_floats(s)?;
    let (s, _) = multispace1(s)?;

    let (s, _) = tag("outer loop")(s)?;
    let (s, _) = multispace1(s)?;

    let (s, v1) = vertex(s)?;
    let (s, v2) = vertex(s)?;
    let (s, v3) = vertex(s)?;

    let (s, _) = tag("endloop")(s)?;
    let (s, _) = multispace1(s)?;

    let (s, _) = tag("endfacet")(s)?;

    Ok((
        s,
        Triangle {
            normal: N::from(normal),
            vertices: [V::from(v1), V::from(v2), V::from(v3)],
        },
    ))
}

fn vertex(s: &[u8]) -> IResult<&[u8], [f32; 3]> {
    let (s, _) = recognize_vertex(s)?;
    let (s, _) = multispace1(s)?;
    let (s, v) = three_floats(s)?;
    let (s, _) = multispace1(s)?;
    Ok((s, v))
}

fn recognize_vertex(s: &[u8]) -> IResult<&[u8], ()> {
    let (s, _) = tag("vertex")(s)?;
    Ok((s, ()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    #[test]
    fn parses_both_ascii_and_binary() {
        // derived from: https://www.thingiverse.com/thing:1187833
        let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER.stl").unwrap();
        let mut moon = BufReader::new(&moon_file);
        let ascii_mesh: Mesh<[f32; 3], [f32; 3]> = parse_stl(&mut moon).unwrap();

        assert_eq!(3698, ascii_mesh.triangles.len());

        // credit: https://www.thingiverse.com/thing:26227
        let vase_file = std::fs::File::open("./fixtures/Root_Vase.stl").unwrap();
        let mut root_vase = BufReader::new(&vase_file);
        let binary_mesh: Mesh<[f32; 3], [f32; 3]> = parse_stl(&mut root_vase).unwrap();

        assert_eq!(596_736, binary_mesh.triangles.len());
    }

    #[test]
    fn parses_ascii_triangles() {
        let triangle_string = "facet normal 0.642777 -2.54044e-006 0.766053
               outer loop
                 vertex 8.08661 0.373289 54.1924
                 vertex 8.02181 0.689748 54.2468
                 vertex 8.10936 0 54.1733
               endloop
             endfacet";

        let triangle = triangle_ascii(triangle_string.as_bytes());

        let test_triangle = Triangle {
            normal: [0.642777, -0.00000254044, 0.766053],
            vertices: [
                [8.08661, 0.373289, 54.1924],
                [8.02181, 0.689748, 54.2468],
                [8.10936, 0.0, 54.1733],
            ],
        };

        assert_eq!(triangle, Ok((vec!().as_slice(), test_triangle)))
    }

    #[test]
    fn parses_ascii_mesh() {
        let mesh_string = "solid OpenSCAD_Model
               facet normal 0.642777 -2.54044e-006 0.766053
                 outer loop
                   vertex 8.08661 0.373289 54.1924
                   vertex 8.02181 0.689748 54.2468
                   vertex 8.10936 0 54.1733
                 endloop
               endfacet
               facet normal -0.281083 -0.678599 -0.678599
                 outer loop
                   vertex -0.196076 7.34845 8.72767
                   vertex 0 8.11983 7.87508
                   vertex 0 7.342 8.6529
                 endloop
               endfacet
             endsolid OpenSCAD_Model";

        let mesh = parse_stl(&mut std::io::Cursor::new(mesh_string.as_bytes().to_owned()));

        let v1 = [
            [8.08661, 0.373289, 54.1924],
            [8.02181, 0.689748, 54.2468],
            [8.10936, 0.0, 54.1733],
        ];

        let v2 = [
            [-0.196076, 7.34845, 8.72767],
            [0.0, 8.11983, 7.87508],
            [0.0, 7.342, 8.6529],
        ];

        let test_mesh = Mesh::new(vec![
            Triangle::new([0.642777, -0.00000254044, 0.766053], v1),
            Triangle::new([-0.281083, -0.678599, -0.678599], v2),
        ]);

        assert_eq!(mesh.unwrap(), test_mesh);
    }

    #[test]
    fn does_ascii_from_file() {
        // derived from: https://www.thingiverse.com/thing:1187833
        let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER.stl").unwrap();
        let mut moon = BufReader::new(&moon_file);
        let mesh: Mesh<[f32; 3], [f32; 3]> = parse_stl(&mut moon).unwrap();

        assert_eq!(3698, mesh.triangles.len());
    }

    #[test]
    fn parses_triangles() {
        let normal = [1.0f32, 7.0f32, 3.0f32];
        let v1 = [0f32, 22.100001f32, 4.1f32];
        let v2 = [1.1f32, 9.10f32, 3.9f32];
        let v3 = [2.0f32, 1.01f32, -5.2f32];

        let normal_bytes: [u8; 12] = unsafe { std::mem::transmute(normal) };
        let v1_bytes: [u8; 12] = unsafe { std::mem::transmute(v1) };
        let v2_bytes: [u8; 12] = unsafe { std::mem::transmute(v2) };
        let v3_bytes: [u8; 12] = unsafe { std::mem::transmute(v3) };

        // a 2-byte short that's ignored
        let attribute_byte_count_bytes: &[u8] = &[0, 0];

        let triangle_bytes = &[
            &normal_bytes,
            &v1_bytes,
            &v2_bytes,
            &v3_bytes,
            attribute_byte_count_bytes,
        ]
        .concat();
        let test_triangle = Triangle {
            normal,
            vertices: [v1, v2, v3],
        };

        assert_eq!(
            triangle_binary(triangle_bytes),
            Ok((vec!().as_slice(), test_triangle))
        );
    }

    #[test]
    fn parses_mesh() {
        let header = vec![0; 80];
        let count = 2u32.to_le_bytes().to_vec();
        let body = vec![
            // triangle 1
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // normal
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // v1
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // v2
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // v3
            0, 0, // uint16
            // triangle 2
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // normal
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // v1
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // v2
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // v3
            0, 0, // uint16
        ];

        let mut all = std::io::Cursor::new(vec![header, count, body].concat());

        let test_mesh = parse_stl(&mut all).unwrap();

        assert_eq!(
            test_mesh,
            Mesh::new(vec![
                Triangle {
                    normal: [0.0, 0.0, 0.0],
                    vertices: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                },
                Triangle {
                    normal: [0.0, 0.0, 0.0],
                    vertices: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                },
            ],)
        );
    }

    #[test]
    fn does_binary_from_file() {
        // credit: https://www.thingiverse.com/thing:26227
        let file = std::fs::File::open("./fixtures/Root_Vase.stl").unwrap();
        let mut root_vase = BufReader::new(&file);
        let start = std::time::Instant::now();
        let mesh: Mesh<[f32; 3], [f32; 3]> = parse_stl(&mut root_vase).unwrap();
        let end = std::time::Instant::now();
        println!("root_vase time: {:?}", end - start);

        assert_eq!(mesh.triangles.len(), 596_736);
    }

    #[test]
    fn does_binary_from_file_starting_with_solid() {
        // credit: https://www.thingiverse.com/thing:26227
        let file = std::fs::File::open("./fixtures/Root_Vase_solid_start.stl").unwrap();
        let mut root_vase = BufReader::new(&file);
        let mesh: Mesh<[f32; 3], [f32; 3]> = parse_stl(&mut root_vase).unwrap();

        assert_eq!(mesh.triangles.len(), 596_736);
    }

    #[test]
    fn does_ascii_file_without_a_closing_solid_name() {
        // derived from: https://www.thingiverse.com/thing:1187833
        let moon_file =
            std::fs::File::open("./fixtures/MOON_PRISM_POWER_no_closing_name.stl").unwrap();
        let mut moon = BufReader::new(&moon_file);
        let result: Mesh<[f32; 3], [f32; 3]> = parse_stl(&mut moon).unwrap();
        assert_eq!(result.triangles.len(), 3698);
    }

    #[test]
    fn parses_stl_with_dos_line_endings_crlf() {
        // derived from: https://www.thingiverse.com/thing:1187833

        let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER_dos.stl").unwrap();
        let mut moon = BufReader::new(&moon_file);
        let result: Mesh<[f32; 3], [f32; 3]> = parse_stl(&mut moon).unwrap();
        assert_eq!(result.triangles.len(), 3698);
    }
}

#[cfg(test)]
mod properties {
    use super::*;
    use quickcheck::*;
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn prop_parses_binary_stl_with_at_least_one_triangle() {
        fn parses_binary_stl_with_at_least_one_triangle(xs: Vec<u8>) -> TestResult {
            // 150 is the length of the 80 byte header plus 50 bytes for a single triangle
            // this may result in partial parses,
            // but this is a first pass at property testing.
            // please remove this comment when this is a bit more robust.
            if xs.len() < 150 {
                return TestResult::discard();
            }

            TestResult::from_bool(
                parse_stl::<std::io::Cursor<Vec<u8>>, [f32; 3], [f32; 3]>(
                    &mut std::io::Cursor::new(xs),
                )
                .is_ok(),
            )
        }

        let mut qc = QuickCheck::new();
        qc.quickcheck(parses_binary_stl_with_at_least_one_triangle as fn(Vec<u8>) -> TestResult);
        qc.min_tests_passed(200);
    }
}
