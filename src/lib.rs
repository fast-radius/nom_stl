//! An parser for binary and ASCII STL files.
//!
//! ## Example
//! ```rust
//! use std::io::BufReader;
//! use std::fs::File;
//! let file = File::open("./fixtures/Root_Vase.stl").unwrap();
//! let mut root_vase = BufReader::new(&file);
//! let mesh: nom_stl::Mesh = nom_stl::parse_stl(&mut root_vase).unwrap();
//! assert_eq!(mesh.triangles().len(), 596_736);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use nom::bytes::complete::{tag, take, take_while1};
use nom::character::complete::{line_ending, multispace0, multispace1};
use nom::combinator::{opt, rest};
use nom::multi::many1;
use nom::number::complete::{float, le_f32};
use nom::IResult;
use std::{
    collections::{HashMap, HashSet},
    convert::TryInto,
    io::{Read, Seek, SeekFrom},
};

type Result<T> = std::result::Result<T, Error>;
type Vertex = [f32; 3];

/// An error is either an IOError (wrapping std::io::Error),
/// or a parse error, indicating that the parser is unable to
/// make progress on an invalid input. This error is derived
/// from the underlying nom_stl error
#[derive(Debug)]
pub enum Error {
    /// A wrapper for a std::io::Error
    IOError(std::io::Error),
    /// Expressing the underlying nom_stl error
    ParseError(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        let e = self;
        Some(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::IOError(error)
    }
}

impl<E: std::fmt::Debug> From<nom::Err<E>> for Error {
    fn from(error: nom::Err<E>) -> Self {
        Error::ParseError(format!("{}", error))
    }
}

/// A triangle type with an included normal vertex and vertices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triangle {
    normal: Vertex,
    vertices: [Vertex; 3],
}

impl Triangle {
    /// Create a new triangle.
    pub fn new(normal: Vertex, vertices: [Vertex; 3]) -> Self {
        Triangle { normal, vertices }
    }

    /// Return the normal vertex of the triangle.
    /// This indicates the "front" of the triangle.
    pub fn normal(&self) -> Vertex {
        self.normal
    }

    /// Return an array of the triangle's corner vertices
    pub fn vertices(&self) -> [Vertex; 3] {
        self.vertices
    }

    /// The size of the `Triangle` struct at runtime.
    pub fn size_of(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// A triangle mesh represented as a vector of `Triangle`.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    triangles: Vec<Triangle>,
}

impl Mesh {
    /// Create a triangle mesh from a `Vec` of `Triangle`.
    pub fn new(triangles: Vec<Triangle>) -> Self {
        Self { triangles }
    }

    /// Return a slice of the mesh's triangles
    pub fn triangles(&self) -> &[Triangle] {
        self.triangles.as_slice()
    }

    /// Returns the an iterator of vertices of all triangles.
    /// This function clones/copies every vertex, and does not deduplicate vertices.
    pub fn vertices(&self) -> impl Iterator<Item = Vertex> + '_ {
        self.vertices_ref().cloned()
    }

    /// Returns an iterator of vertex references for all triangles.
    /// Does not deduplicate any vertices.
    pub fn vertices_ref(&self) -> impl Iterator<Item = &Vertex> {
        self.triangles()
            .iter()
            .flat_map(|triangle| triangle.vertices.as_ref())
    }

    /// Returns an iterator of all unique vertices in the mesh.
    /// This function clones/copies every vertex.
    pub fn unique_vertices(&self) -> impl Iterator<Item = Vertex> {
        let set = self
            .vertices_ref()
            .map(|vertex| {
                [
                    vertex[0].to_bits(),
                    vertex[1].to_bits(),
                    vertex[2].to_bits(),
                ]
            })
            .collect::<HashSet<_>>();

        set.into_iter()
            .map(|[x, y, z]| [f32::from_bits(x), f32::from_bits(y), f32::from_bits(z)])
    }

    /// The size of the `Mesh` struct at runtime.
    pub fn size_of(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// A triangle mesh represented as a vector of `IndexTriangle`
/// and a vector of `Vertex`.
pub struct IndexMesh {
    triangles: Vec<IndexTriangle>,
    vertices: Vec<Vertex>,
}

impl IndexMesh {
    /// Returns a slice of all `IndexTriangle` in the mesh.
    pub fn triangles(&self) -> &[IndexTriangle] {
        self.triangles.as_slice()
    }

    /// Returns a slice of all vertices in the mesh.
    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    /// The size of the `IndexMesh` struct at runtime.
    pub fn size_of(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// A triangle type which contains a normal vertex and index references
/// to vertices contained in a separate vertices container.
/// See `IndexMesh`.
pub struct IndexTriangle {
    normal: Vertex,
    vertices_indices: [usize; 3],
}

impl IndexTriangle {
    /// The normal vector.
    pub fn normal(&self) -> Vertex {
        self.normal
    }

    /// Returns the vertices for the `IndexTriangle` by looking them up
    /// in the given `vertices` slice.
    pub fn vertices(&self, vertices: &[Vertex]) -> [Vertex; 3] {
        [
            vertices[self.vertices_indices[0]],
            vertices[self.vertices_indices[1]],
            vertices[self.vertices_indices[2]],
        ]
    }

    /// Returns the positions of the triangle's 3 vertices
    /// in the separate vertices container.
    pub fn vertices_indices(&self) -> [usize; 3] {
        self.vertices_indices
    }

    /// The size of the `IndexTriangle` at runtime.
    pub fn size_of(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl From<Mesh> for IndexMesh {
    fn from(mesh: Mesh) -> Self {
        let mut vertices: Vec<[f32; 3]> = vec![];
        let mut vertices_bits_to_indices: HashMap<[u32; 3], usize> = HashMap::new();
        let mut vertices_indices: [usize; 3] = [0, 0, 0];

        let index_triangles = mesh
            .triangles
            .iter()
            .map(|triangle| {
                for (i, vertex) in triangle.vertices.iter().enumerate() {
                    let bits = [
                        vertex[0].to_bits(),
                        vertex[1].to_bits(),
                        vertex[2].to_bits(),
                    ];

                    if let Some(index) = vertices_bits_to_indices.get(&bits) {
                        vertices_indices[i] = *index;
                    } else {
                        let index = vertices.len();
                        vertices_bits_to_indices.insert(bits, index);
                        vertices_indices[i] = index;
                        vertices.push(*vertex);
                    }
                }

                IndexTriangle {
                    normal: triangle.normal,
                    vertices_indices,
                }
            })
            .collect();

        IndexMesh {
            triangles: index_triangles,
            vertices,
        }
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
pub fn parse_stl<R: Read + Seek>(bytes: &mut R) -> Result<Mesh> {
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

fn mesh_binary<R: Read>(mut s: R) -> Result<Mesh> {
    let mut header_and_triangles_count = [0u8; HEADER_SIZE_BYTES];

    s.read_exact(&mut header_and_triangles_count)?;

    let reported_triangle_count = u32::from_le_bytes(
        header_and_triangles_count[80..84]
            .try_into()
            .expect("Could not get four bytes to create u32"),
    );

    // previously we optimized this with `Vec::with_capacity`, but
    // fuzzing with afl++ uncovered that it is possible to crash the process
    // by passing a specially crafted stl with a triangle count value larger
    // than the system memory
    let mut all_triangles: Vec<Triangle> = Vec::new();

    let triangles_reader: TrianglesIter<R> =
        TrianglesIter::new(s, reported_triangle_count as usize);

    for triangle in triangles_reader {
        all_triangles.push(triangle?);
    }

    let mesh = Mesh::new(all_triangles);

    Ok(mesh)
}
#[derive(Debug)]
struct TrianglesIter<R: Read> {
    reader: R,
    buf: Vec<u8>,
    triangles_to_read: usize,
    triangles_read: usize,
}

impl<R: Read> TrianglesIter<R> {
    fn new(reader: R, triangles_to_read: usize) -> Self {
        TrianglesIter {
            reader,
            buf: vec![0u8; TRIANGLE_SIZE_BYTES],
            triangles_to_read,
            triangles_read: 0,
        }
    }
}

impl<R: Read> Iterator for TrianglesIter<R> {
    type Item = Result<Triangle>;

    fn next(&mut self) -> Option<Result<Triangle>> {
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

fn three_f32s(s: &[u8]) -> IResult<&[u8], Vertex> {
    assert!(s.len() >= 12);
    let (s, f1) = le_f32(s)?;
    let (s, f2) = le_f32(s)?;
    let (s, f3) = le_f32(s)?;

    Ok((s, [f1, f2, f3]))
}

fn triangle_binary(s: &[u8]) -> IResult<&[u8], Triangle> {
    let (s, normal) = three_f32s(s)?;
    let (s, v1) = three_f32s(s)?;
    let (s, v2) = three_f32s(s)?;
    let (s, v3) = three_f32s(s)?;
    let (s, _attribute_byte_count) = take(2usize)(s)?;

    Ok((
        s,
        Triangle {
            normal,
            vertices: [v1, v2, v3],
        },
    ))
}

// ASCII GRAMMAR
/////////////////////////////////////////////////////////////////

fn not_line_ending(c: u8) -> bool {
    c != b'\r' && c != b'\n'
}

fn mesh_ascii(s: &[u8]) -> Result<Mesh> {
    let (s, _) = tag("solid")(s).map_err(to_crate_err)?;
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

fn three_floats(s: &[u8]) -> IResult<&[u8], Vertex> {
    let (s, f1) = float(s)?;
    let (s, _) = multispace1(s)?;
    let (s, f2) = float(s)?;
    let (s, _) = multispace1(s)?;
    let (s, f3) = float(s)?;

    Ok((s, [f1, f2, f3]))
}

fn triangle_ascii(s: &[u8]) -> IResult<&[u8], Triangle> {
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
            normal,
            vertices: [v1, v2, v3],
        },
    ))
}

fn vertex(s: &[u8]) -> IResult<&[u8], Vertex> {
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

    fn to_bytes(vec: [f32; 3]) -> [u8; 12] {
        [
            vec[0].to_le_bytes()[0],
            vec[0].to_le_bytes()[1],
            vec[0].to_le_bytes()[2],
            vec[0].to_le_bytes()[3],
            vec[1].to_le_bytes()[0],
            vec[1].to_le_bytes()[1],
            vec[1].to_le_bytes()[2],
            vec[1].to_le_bytes()[3],
            vec[2].to_le_bytes()[0],
            vec[2].to_le_bytes()[1],
            vec[2].to_le_bytes()[2],
            vec[2].to_le_bytes()[3],
        ]
    }

    #[test]
    fn parses_both_ascii_and_binary() {
        // derived from: https://www.thingiverse.com/thing:1187833
        let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER.stl").unwrap();
        let mut moon = BufReader::new(&moon_file);
        let ascii_mesh: Mesh = parse_stl(&mut moon).unwrap();

        assert_eq!(3698, ascii_mesh.triangles.len());

        // credit: https://www.thingiverse.com/thing:26227
        let vase_file = std::fs::File::open("./fixtures/Root_Vase.stl").unwrap();
        let mut root_vase = BufReader::new(&vase_file);
        let binary_mesh: Mesh = parse_stl(&mut root_vase).unwrap();

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
        let mesh: Mesh = parse_stl(&mut moon).unwrap();

        assert_eq!(3698, mesh.triangles.len());
    }

    #[test]
    fn parses_triangles() {
        let normal = [1.0f32, 7.0f32, 3.0f32];
        let v1 = [0f32, 22.100001f32, 4.1f32];
        let v2 = [1.1f32, 9.10f32, 3.9f32];
        let v3 = [2.0f32, 1.01f32, -5.2f32];

        let normal_bytes = to_bytes(normal);
        let v1_bytes: [u8; 12] = to_bytes(v1);
        let v2_bytes: [u8; 12] = to_bytes(v2);
        let v3_bytes: [u8; 12] = to_bytes(v3);

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
        let mesh: Mesh = parse_stl(&mut root_vase).unwrap();
        let end = std::time::Instant::now();
        println!("root_vase time: {:?}", end - start);

        assert_eq!(mesh.triangles.len(), 596_736);
    }

    #[test]
    fn does_binary_from_file_starting_with_solid() {
        // credit: https://www.thingiverse.com/thing:26227
        let file = std::fs::File::open("./fixtures/Root_Vase_solid_start.stl").unwrap();
        let mut root_vase = BufReader::new(&file);
        let mesh: Mesh = parse_stl(&mut root_vase).unwrap();

        assert_eq!(mesh.triangles.len(), 596_736);
    }

    #[test]
    fn does_ascii_file_without_a_closing_solid_name() {
        // derived from: https://www.thingiverse.com/thing:1187833
        let moon_file =
            std::fs::File::open("./fixtures/MOON_PRISM_POWER_no_closing_name.stl").unwrap();
        let mut moon = BufReader::new(&moon_file);
        let result: Mesh = parse_stl(&mut moon).unwrap();
        assert_eq!(result.triangles.len(), 3698);
    }

    #[test]
    fn parses_stl_with_dos_line_endings_crlf() {
        // derived from: https://www.thingiverse.com/thing:1187833

        let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER_dos.stl").unwrap();
        let mut moon = BufReader::new(&moon_file);
        let result: Mesh = parse_stl(&mut moon).unwrap();
        assert_eq!(result.triangles.len(), 3698);
    }

    #[test]
    fn all_vertices() {
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
               facet normal -0.281083 -0.678599 -0.678599
                 outer loop
                   vertex 8.08661 0.373289 54.1924
                   vertex 8.02181 0.689748 54.2468
                   vertex 0 7.342 8.6529
                 endloop
               endfacet
             endsolid OpenSCAD_Model";

        let mesh = parse_stl(&mut std::io::Cursor::new(mesh_string.as_bytes().to_owned())).unwrap();

        assert_eq!(
            mesh.vertices_ref().collect::<Vec<&Vertex>>().len(),
            mesh.triangles().len() * 3
        );

        assert_eq!(
            mesh.vertices().collect::<Vec<Vertex>>().len(),
            mesh.triangles().len() * 3
        );
    }

    #[test]
    fn makes_unique_vertices() {
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
               facet normal -0.281083 -0.678599 -0.678599
                 outer loop
                   vertex 8.08661 0.373289 54.1924
                   vertex 8.02181 0.689748 54.2468
                   vertex 0 7.342 8.6529
                 endloop
               endfacet
             endsolid OpenSCAD_Model";

        let mesh = parse_stl(&mut std::io::Cursor::new(mesh_string.as_bytes().to_owned())).unwrap();

        assert_eq!(mesh.unique_vertices().collect::<Vec<_>>().len(), 6);
    }

    #[test]
    fn creates_an_index_mesh() {
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
                   vertex 8.08661 0.373289 54.1924
                   vertex 8.02181 0.689748 54.2468
                   vertex 0 7.342 8.6529
                 endloop
               endfacet
               facet normal -0.281083 -0.678599 -0.678599
                 outer loop
                   vertex 8.08661 0.373289 54.1924
                   vertex 8.02181 0.689748 54.2468
                   vertex 4.0 4.0 4.0
                 endloop
               endfacet
             endsolid OpenSCAD_Model";

        let mesh = parse_stl(&mut std::io::Cursor::new(mesh_string.as_bytes().to_owned())).unwrap();

        let index_mesh: IndexMesh = mesh.into();

        assert_eq!(index_mesh.triangles().len(), 3);
        assert_eq!(index_mesh.vertices().len(), 5);
    }

    #[test]
    fn ascii_without_an_opening_file_name() {
        let mesh_string = "solid
               facet normal 0.642777 -2.54044e-006 0.766053
                 outer loop
                   vertex 8.08661 0.373289 54.1924
                   vertex 8.02181 0.689748 54.2468
                   vertex 8.10936 0 54.1733
                 endloop
               endfacet
               facet normal -0.281083 -0.678599 -0.678599
                 outer loop
                   vertex 8.08661 0.373289 54.1924
                   vertex 8.02181 0.689748 54.2468
                   vertex 0 7.342 8.6529
                 endloop
               endfacet
               facet normal -0.281083 -0.678599 -0.678599
                 outer loop
                   vertex 8.08661 0.373289 54.1924
                   vertex 8.02181 0.689748 54.2468
                   vertex 4.0 4.0 4.0
                 endloop
               endfacet
             endsolid OpenSCAD_Model";

        let mesh = parse_stl(&mut std::io::Cursor::new(mesh_string.as_bytes().to_owned())).unwrap();

        let index_mesh: IndexMesh = mesh.into();

        assert_eq!(index_mesh.triangles().len(), 3);
        assert_eq!(index_mesh.vertices().len(), 5);
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
                parse_stl::<std::io::Cursor<Vec<u8>>>(&mut std::io::Cursor::new(xs)).is_ok(),
            )
        }

        let mut qc = QuickCheck::new();
        qc.quickcheck(parses_binary_stl_with_at_least_one_triangle as fn(Vec<u8>) -> TestResult);
        qc.min_tests_passed(200);
    }
}
