use nom::bytes::complete::{tag, take, take_while1};
use nom::character::complete::{line_ending, multispace0, multispace1};
use nom::combinator::{complete, opt, rest};
use nom::multi::many1;
use nom::number::complete::{float, le_f32};
use nom::{eat_separator, named, IResult};
use std::error::Error;
use std::io::{Read, Seek, SeekFrom};
use std::marker::PhantomData;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[cfg(feature = "fx")]
use rustc_hash::FxHashMap as HashMap;
#[cfg(not(feature = "fx"))]
use std::collections::HashMap;

type Index = usize;

pub trait XYZ: Clone + Copy + From<[f32; 3]> + Into<[f32; 3]> {}

impl XYZ for [f32; 3] {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnindexedTriangle<N, V> {
    normal: N,
    vertices: [V; 3],
}

impl<N: XYZ, V: XYZ> UnindexedTriangle<N, V> {
    pub fn new(normal: N, vertices: [V; 3]) -> Self {
        Self { normal, vertices }
    }

    #[inline]
    pub fn normal(&self) -> N {
        self.normal
    }

    #[inline]
    pub fn vertices(&self) -> [V; 3] {
        self.vertices
    }

    pub fn size_of(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IndexedTriangle<N, V> {
    normal: N,
    vertices: [Index; 3],
    _marker: PhantomData<V>,
}

impl<N: XYZ, V: XYZ> IndexedTriangle<N, V> {
    pub fn new(normal: N, vertices: [Index; 3]) -> Self {
        Self {
            normal,
            vertices,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn normal(&self) -> N {
        self.normal
    }

    pub fn vertices(&self, parent: &IndexedMesh<N, V>) -> [V; 3] {
        [
            parent.vertices[self.vertices[0]],
            parent.vertices[self.vertices[1]],
            parent.vertices[self.vertices[2]],
        ]
    }

    pub fn size_of(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnindexedMesh<N: XYZ, V: XYZ> {
    triangles: Vec<UnindexedTriangle<N, V>>,
}

impl<N: XYZ, V: XYZ> UnindexedMesh<N, V> {
    pub fn triangles(&self) -> &[UnindexedTriangle<N, V>] {
        self.triangles.as_slice()
    }

    pub fn size_of(&self) -> usize {
        let struct_size = std::mem::size_of::<Self>();
        let triangles_size = self.triangles.len() * std::mem::size_of::<UnindexedTriangle<N, V>>();
        struct_size + triangles_size
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IndexedMesh<N: XYZ, V: XYZ> {
    vertices: Vec<V>,
    triangles: Vec<IndexedTriangle<N, V>>,
}

impl<N: XYZ, V: XYZ> IndexedMesh<N, V> {
    pub fn triangles(&self) -> &[IndexedTriangle<N, V>] {
        self.triangles.as_slice()
    }

    pub fn size_of(&self) -> usize {
        let struct_size = std::mem::size_of::<Self>();
        let triangle_size = std::mem::size_of::<IndexedTriangle<N, V>>();
        let vertices_size = self.vertices.len() * std::mem::size_of::<V>();
        let triangles_size = self.triangles.len() * triangle_size;

        struct_size + vertices_size + triangles_size
    }
}

impl<N: XYZ, V: XYZ> From<UnindexedMesh<N, V>> for IndexedMesh<N, V> {
    fn from(mesh: UnindexedMesh<N, V>) -> Self {
        build_indexed_mesh(mesh.triangles)
    }
}

impl<N: XYZ, V: XYZ> From<IndexedMesh<N, V>> for UnindexedMesh<N, V> {
    fn from(indexed_mesh: IndexedMesh<N, V>) -> Self {
        let mut triangles = Vec::with_capacity(indexed_mesh.triangles.len());
        for indexed_triangle in indexed_mesh.triangles {
            let triangle = UnindexedTriangle {
                normal: indexed_triangle.normal,
                vertices: [
                    indexed_mesh.vertices[indexed_triangle.vertices[0]],
                    indexed_mesh.vertices[indexed_triangle.vertices[1]],
                    indexed_mesh.vertices[indexed_triangle.vertices[2]],
                ],
            };

            triangles.push(triangle)
        }

        Self { triangles }
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
pub fn parse_stl_indexed<R: Read + Seek, N: XYZ, V: XYZ>(
    bytes: &mut R,
) -> Result<(Vec<u8>, IndexedMesh<N, V>)> {
    if contains_facet_normal_bytes(bytes.by_ref()) {
        bytes.seek(SeekFrom::Start(0))?;

        let mut buf = vec![];

        bytes.read_to_end(&mut buf)?;

        let res = indexed_mesh_ascii(&buf);

        match res {
            Ok((s, mesh)) => Ok((s, mesh)),
            Err(e) => Err(e),
        }
    } else {
        bytes.seek(SeekFrom::Start(0))?;
        indexed_mesh_binary(bytes.by_ref())
    }
}

pub fn parse_stl_unindexed<R: Read + Seek, N: XYZ, V: XYZ>(
    bytes: &mut R,
) -> Result<(Vec<u8>, UnindexedMesh<N, V>)> {
    if contains_facet_normal_bytes(bytes.by_ref()) {
        bytes.seek(SeekFrom::Start(0))?;

        let mut buf = vec![];

        bytes.read_to_end(&mut buf)?;

        let res = mesh_ascii(&buf);

        match res {
            Ok((s, mesh)) => Ok((s, mesh)),
            Err(e) => Err(e),
        }
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
const NOMINAL_CHUNK_SIZE: usize = 100;
const NOMINAL_CHUNK_SIZE_BYTES: usize = TRIANGLE_SIZE_BYTES * NOMINAL_CHUNK_SIZE; // 50 * 100 = 5,000 bytes

fn indexed_mesh_binary<'a, R: Read, N: 'a + XYZ, V: 'a + XYZ>(
    s: &mut R,
) -> Result<(Vec<u8>, IndexedMesh<N, V>)> {
    let (s, mesh) = mesh_binary(s)?;
    let indexed_mesh: IndexedMesh<N, V> = mesh.into();

    Ok((s, indexed_mesh))
}

fn mesh_binary<R: Read, N: XYZ, V: XYZ>(s: &mut R) -> Result<(Vec<u8>, UnindexedMesh<N, V>)> {
    let mut header_and_triangles_count = vec![0u8; HEADER_SIZE_BYTES];

    let read_result = s.read_exact(&mut header_and_triangles_count);
    match read_result {
        Ok(()) => (),
        Err(e) => return Err(Box::new(e)),
    }

    let reported_triangle_count = u32::from_le_bytes([
        header_and_triangles_count[80],
        header_and_triangles_count[81],
        header_and_triangles_count[82],
        header_and_triangles_count[83],
    ]);

    let calculated_bytes_size = reported_triangle_count as usize * TRIANGLE_SIZE_BYTES;

    let number_of_nominal_chunks_to_read = calculated_bytes_size / NOMINAL_CHUNK_SIZE_BYTES;

    let remainder_chunk_length =
        calculated_bytes_size - (number_of_nominal_chunks_to_read * NOMINAL_CHUNK_SIZE_BYTES);

    let mut all_triangles: Vec<UnindexedTriangle<N, V>> =
        Vec::with_capacity(reported_triangle_count as usize);

    let mut bytes_read: usize = 0;

    let mut chunk_buf = vec![0u8; NOMINAL_CHUNK_SIZE_BYTES];

    for _chunk in 0..number_of_nominal_chunks_to_read {
        {
            let mut chunk_buf = &mut chunk_buf;
            let read_result = s.read_exact(&mut chunk_buf);

            match read_result {
                Ok(()) => {
                    bytes_read += NOMINAL_CHUNK_SIZE_BYTES;
                }
                Err(e) => return Err(Box::new(e)),
            }
        };

        let triangles_result =
            nom::multi::count(complete(triangle_binary), NOMINAL_CHUNK_SIZE)(&chunk_buf);

        match triangles_result {
            Ok((_, mut triangles)) => {
                all_triangles.append(&mut triangles);
            }
            Err(e) => return Err(Box::new(e.to_owned())),
        }
    }

    if remainder_chunk_length > 0 {
        let mut chunk_buf = Vec::with_capacity(remainder_chunk_length);
        let read_result = s.read_to_end(&mut chunk_buf);

        match read_result {
            Ok(chunk_bytes_read) => {
                bytes_read += chunk_bytes_read;
            }
            Err(e) => return Err(Box::new(e)),
        }

        let triangles_result = nom::multi::many1(complete(triangle_binary))(&chunk_buf);

        match triangles_result {
            Ok((_, mut triangles)) => {
                all_triangles.append(&mut triangles);
            }
            Err(e) => return Err(Box::new(e.to_owned())),
        }
    }

    let rem = if bytes_read < calculated_bytes_size {
        let mut rem = vec![];
        let read_result = s.read_to_end(&mut rem);

        match read_result {
            Ok(_) => rem,
            Err(e) => return Err(Box::new(e)),
        }
    } else {
        vec![]
    };

    let mesh = UnindexedMesh {
        triangles: all_triangles,
    };

    Ok((rem, mesh))
}

fn three_f32s(s: &[u8]) -> IResult<&[u8], [f32; 3]> {
    let (s, f1) = le_f32(s)?;
    let (s, f2) = le_f32(s)?;
    let (s, f3) = le_f32(s)?;

    Ok((s, [f1, f2, f3]))
}

fn triangle_binary<N: XYZ, V: XYZ>(s: &[u8]) -> IResult<&[u8], UnindexedTriangle<N, V>> {
    let (s, normal) = three_f32s(s)?;
    let (s, v1) = three_f32s(s)?;
    let (s, v2) = three_f32s(s)?;
    let (s, v3) = three_f32s(s)?;
    let (s, _attribute_byte_count) = take(2usize)(s)?;

    Ok((
        s,
        UnindexedTriangle {
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

type BytesSliceResult<'a> = IResult<&'a [u8], &'a [u8]>;

fn indexed_mesh_ascii<N: XYZ, V: XYZ>(s: &[u8]) -> Result<(Vec<u8>, IndexedMesh<N, V>)> {
    let (s, mesh) = mesh_ascii(s)?;
    let indexed_mesh = mesh.into();

    Ok((s.to_vec(), indexed_mesh))
}

fn mesh_ascii<'a, N: XYZ, V: XYZ>(s: &'a [u8]) -> Result<(Vec<u8>, UnindexedMesh<N, V>)> {
    let res: BytesSliceResult<'a> = tag("solid ")(s);

    let (s, _): (&'a [u8], ()) = match res {
        Ok((s, _)) => (s, ()),
        Err(e) => return Err(Box::new(e.to_owned())),
    };

    let res: IResult<&'a [u8], Option<&[u8]>> = opt(take_while1(not_line_ending))(s);

    let (s, _): (&'a [u8], ()) = match res {
        Ok((s, _)) => (s, ()),
        Err(e) => return Err(Box::new(e.to_owned())),
    };

    let res: BytesSliceResult<'a> = line_ending(s);

    let (s, _): (&'a [u8], ()) = match res {
        Ok((s, _)) => (s, ()),
        Err(e) => return Err(Box::new(e.to_owned())),
    };

    let res: IResult<&'a [u8], Vec<UnindexedTriangle<N, V>>> = many1(triangle_ascii)(s);

    let (s, triangles): (&'a [u8], Vec<UnindexedTriangle<N, V>>) = match res {
        Ok((s, triangles)) => (s, triangles),
        Err(e) => return Err(Box::new(e.to_owned())),
    };

    let res: BytesSliceResult<'a> = multispace1(s);

    let (s, _): (&'a [u8], ()) = match res {
        Ok((s, _)) => (s, ()),
        Err(e) => return Err(Box::new(e.to_owned())),
    };

    let res: BytesSliceResult<'a> = tag("endsolid")(s);

    let (s, _): (&'a [u8], ()) = match res {
        Ok((s, _)) => (s, ()),
        Err(e) => return Err(Box::new(e.to_owned())),
    };

    let res: IResult<&'a [u8], Option<&[u8]>> = opt(rest)(s);

    let (s, _) = match res {
        Ok((s, _)) => (s, ()),
        Err(e) => return Err(Box::new(e.to_owned())),
    };

    let mesh = UnindexedMesh { triangles };

    Ok((s.to_vec(), mesh))
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

fn vertex(s: &[u8]) -> IResult<&[u8], ()> {
    let (s, _) = tag("vertex")(s)?;
    Ok((s, ()))
}

fn triangle_ascii<N: XYZ, V: XYZ>(s: &[u8]) -> IResult<&[u8], UnindexedTriangle<N, V>> {
    let (s, _) = multispace0(s)?;
    let (s, _) = tag("facet normal")(s)?;
    let (s, _) = multispace1(s)?;

    let (s, normal) = three_floats(s)?;
    let (s, _) = multispace1(s)?;

    let (s, _) = tag("outer loop")(s)?;
    let (s, _) = multispace1(s)?;

    let (s, _) = vertex(s)?;
    let (s, _) = multispace1(s)?;
    let (s, v1) = three_floats(s)?;
    let (s, _) = multispace1(s)?;

    let (s, _) = vertex(s)?;
    let (s, _) = multispace1(s)?;
    let (s, v2) = three_floats(s)?;
    let (s, _) = multispace1(s)?;

    let (s, _) = vertex(s)?;
    let (s, _) = multispace1(s)?;
    let (s, v3) = three_floats(s)?;
    let (s, _) = multispace1(s)?;

    let (s, _) = tag("endloop")(s)?;
    let (s, _) = multispace1(s)?;

    let (s, _) = tag("endfacet")(s)?;

    Ok((
        s,
        UnindexedTriangle {
            normal: N::from(normal),
            vertices: [V::from(v1), V::from(v2), V::from(v3)],
        },
    ))
}

fn build_indexed_mesh<N: XYZ, V: XYZ>(
    triangles: Vec<UnindexedTriangle<N, V>>,
) -> IndexedMesh<N, V> {
    #[cfg(feature = "fx")]
    let mut indexes = HashMap::default();
    #[cfg(not(feature = "fx"))]
    let mut indexes = HashMap::new();

    let mut vertices = Vec::new();

    let mut indexed_triangles: Vec<IndexedTriangle<N, V>> = Vec::with_capacity(triangles.len());

    for triangle in triangles.into_iter() {
        let mut vertex_indices = [0; 3];

        for (i, vertex) in triangle.vertices.iter().enumerate() {
            let v = *vertex;
            let as_f32s: [f32; 3] = v.into();

            let vertex_as_u32_bits = [
                as_f32s[0].to_bits(),
                as_f32s[1].to_bits(),
                as_f32s[2].to_bits(),
            ];

            let vertices_length = vertices.len();

            let index = *indexes
                .entry(vertex_as_u32_bits)
                .or_insert_with(|| vertices_length);

            if index == vertices_length {
                vertices.push(*vertex);
            }

            vertex_indices[i] = index;
        }

        let indexed_triangle = IndexedTriangle::new(triangle.normal, vertex_indices);

        indexed_triangles.push(indexed_triangle);
    }

    IndexedMesh {
        vertices,
        triangles: indexed_triangles,
    }
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
        let (remaining, ascii_mesh): (Vec<u8>, IndexedMesh<[f32; 3], [f32; 3]>) =
            parse_stl_indexed(&mut moon).unwrap();

        assert_eq!(Vec::<u8>::new(), remaining);
        assert_eq!(3698, ascii_mesh.triangles.len());

        // credit: https://www.thingiverse.com/thing:26227
        let vase_file = std::fs::File::open("./fixtures/Root_Vase.stl").unwrap();
        let mut root_vase = BufReader::new(&vase_file);
        let (remaining, binary_mesh): (Vec<u8>, IndexedMesh<[f32; 3], [f32; 3]>) =
            parse_stl_indexed(&mut root_vase).unwrap();

        assert_eq!(Vec::<u8>::new(), remaining);
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

        let test_triangle = UnindexedTriangle {
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
    fn parses_ascii_indexed_mesh() {
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

        let indexed_mesh =
            parse_stl_indexed(&mut std::io::Cursor::new(mesh_string.as_bytes().to_owned()));

        let vertices = vec![
            [8.08661, 0.373289, 54.1924],
            [8.02181, 0.689748, 54.2468],
            [8.10936, 0.0, 54.1733],
            [-0.196076, 7.34845, 8.72767],
            [0.0, 8.11983, 7.87508],
            [0.0, 7.342, 8.6529],
        ];

        let test_mesh = IndexedMesh {
            vertices,
            triangles: vec![
                IndexedTriangle::new([0.642777, -0.00000254044, 0.766053], [0, 1, 2]),
                IndexedTriangle::new([-0.281083, -0.678599, -0.678599], [3, 4, 5]),
            ],
        };

        assert_eq!(indexed_mesh.unwrap().1, test_mesh);
    }

    #[test]
    fn does_ascii_from_file() {
        // derived from: https://www.thingiverse.com/thing:1187833
        let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER.stl").unwrap();
        let mut moon = BufReader::new(&moon_file);
        let (remaining, mesh): (Vec<u8>, IndexedMesh<[f32; 3], [f32; 3]>) =
            parse_stl_indexed(&mut moon).unwrap();

        assert_eq!(Vec::<u8>::new(), remaining);
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
        let test_triangle = UnindexedTriangle {
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

        let test_mesh = parse_stl_unindexed(&mut all).unwrap();

        assert_eq!(test_mesh.0, vec![].as_slice());

        assert_eq!(
            test_mesh.1,
            UnindexedMesh {
                triangles: vec!(
                    UnindexedTriangle {
                        normal: [0.0, 0.0, 0.0],
                        vertices: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                    },
                    UnindexedTriangle {
                        normal: [0.0, 0.0, 0.0],
                        vertices: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                    },
                ),
            }
        );
    }

    #[test]
    fn parses_indexed_mesh() {
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

        let concated = vec![header, count, body].concat();

        let mut all = std::io::Cursor::new(concated);

        let vertices = vec![[0.0, 0.0, 0.0]];

        assert_eq!(
            indexed_mesh_binary(&mut all).unwrap().1,
            IndexedMesh {
                vertices: vertices.clone(),
                triangles: vec![
                    IndexedTriangle::new([0.0, 0.0, 0.0], [0, 0, 0],),
                    IndexedTriangle::new([0.0, 0.0, 0.0], [0, 0, 0],),
                ],
            }
        );
    }

    #[test]
    fn does_binary_from_file() {
        // credit: https://www.thingiverse.com/thing:26227
        let file = std::fs::File::open("./fixtures/Root_Vase.stl").unwrap();
        let mut root_vase = BufReader::new(&file);
        let start = std::time::Instant::now();
        let (remaining, mesh): (Vec<u8>, IndexedMesh<[f32; 3], [f32; 3]>) =
            parse_stl_indexed(&mut root_vase).unwrap();
        let end = std::time::Instant::now();
        println!("root_vase time: {:?}", end - start);

        assert_eq!(remaining, b"");
        assert_eq!(mesh.triangles.len(), 596_736);
    }

    #[test]
    fn does_binary_from_file_starting_with_solid() {
        // credit: https://www.thingiverse.com/thing:26227
        let file = std::fs::File::open("./fixtures/Root_Vase_solid_start.stl").unwrap();
        let mut root_vase = BufReader::new(&file);
        let (remaining, mesh): (Vec<u8>, IndexedMesh<[f32; 3], [f32; 3]>) =
            parse_stl_indexed(&mut root_vase).unwrap();

        assert_eq!(remaining, b"");
        assert_eq!(mesh.triangles.len(), 596_736);
    }

    #[test]
    fn does_ascii_file_without_a_closing_solid_name() {
        // derived from: https://www.thingiverse.com/thing:1187833
        let moon_file =
            std::fs::File::open("./fixtures/MOON_PRISM_POWER_no_closing_name.stl").unwrap();
        let mut moon = BufReader::new(&moon_file);
        let (remaining, result): (Vec<u8>, IndexedMesh<[f32; 3], [f32; 3]>) =
            parse_stl_indexed(&mut moon).unwrap();
        assert_eq!(remaining, &[]);
        assert_eq!(result.triangles.len(), 3698);
    }

    #[test]
    fn parses_stl_with_dos_line_endings_crlf() {
        // derived from: https://www.thingiverse.com/thing:1187833

        let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER_dos.stl").unwrap();
        let mut moon = BufReader::new(&moon_file);
        let (remaining, result): (Vec<u8>, IndexedMesh<[f32; 3], [f32; 3]>) =
            parse_stl_indexed(&mut moon).unwrap();
        assert!(remaining.is_empty());
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
                parse_stl_indexed::<std::io::Cursor<Vec<u8>>, [f32; 3], [f32; 3]>(
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
