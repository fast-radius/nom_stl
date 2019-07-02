use nom::character::complete::*;
use nom::combinator::rest;
use nom::number::streaming::*;
use nom::*;

#[cfg(feature = "hashbrown")]
use hashbrown::HashMap;
#[cfg(not(feature = "hashbrown"))]
use std::collections::HashMap;

pub type Vertex = [f32; 3];
type Index = usize;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triangle {
    pub normal: Vertex,
    pub vertices: [Vertex; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub reported_count: u32,
    pub triangles: Vec<Triangle>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndexedTriangle {
    pub normal: Vertex,
    pub vertices: [Index; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct IndexedMesh {
    pub reported_triangles_count: u32,
    pub actual_triangles_count: usize,
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<IndexedTriangle>,
}

// BOTH GRAMMARS
/////////////////////////////////////////////////////////////////

type ParseResult<T> = std::result::Result<(T, IndexedMesh), nom::Err<(T, nom::error::ErrorKind)>>;

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
pub fn parse_stl(bytes: &[u8]) -> ParseResult<&[u8]> {
    if contains_facet_normal_bytes(&bytes) {
        indexed_mesh_ascii(bytes)
    } else {
        indexed_mesh_binary(bytes)
    }
}

fn contains_facet_normal_bytes(bytes: &[u8]) -> bool {
    let mut identifier_search_bytes_length =
        match std::env::var("NOM_IDENTIFIER_SEARCH_BYTES_LENGTH") {
            Ok(length) => length.parse().unwrap_or_else(|_| 1024),
            Err(_e) => 1024,
        };

    identifier_search_bytes_length = if bytes.len() <= identifier_search_bytes_length {
        bytes.len()
    } else {
        identifier_search_bytes_length
    };

    search_bytes(&bytes[..identifier_search_bytes_length], b"facet normal").is_some()
}

fn search_bytes(bytes: &[u8], target: &[u8]) -> Option<usize> {
    bytes
        .windows(target.len())
        .position(|window| window == target)
}

// BINARY GRAMMAR
/////////////////////////////////////////////////////////////////

named!(
    indexed_mesh_binary<IndexedMesh>,
    complete!(do_parse!(
        // we do not check for `solid` here,
        // instead checking in `parse_stl`
        take!(80)
            >> reported_count: le_u32
            >> triangles: many1!(complete!(triangle_binary))
            >> (build_indexed_mesh(&triangles, reported_count))
    ))
);

named!(
    pub mesh<Mesh>,
    do_parse!(
        _header: take!(80)
            >> reported_count: le_u32
            >> triangles: many1!(complete!(triangle_binary))
            >> (Mesh {
                reported_count: reported_count,
                triangles: triangles
            })
    )
);

named!(
    three_f32s<[f32; 3]>,
    do_parse!(f1: le_f32 >> f2: le_f32 >> f3: le_f32 >> ([f1, f2, f3]))
);

named!(
    triangle_binary<Triangle>,
    do_parse!(
        normal: three_f32s
            >> v1: three_f32s
            >> v2: three_f32s
            >> v3: three_f32s
            >> _attribute_byte_count: take!(2)
            >> (Triangle {
                normal: normal,
                vertices: [v1, v2, v3]
            })
    )
);

// ASCII GRAMMAR
/////////////////////////////////////////////////////////////////

fn not_line_ending(c: u8) -> bool {
    c != b'\r' && c != b'\n'
}

named!(
    indexed_mesh_ascii<IndexedMesh>,
    do_parse!(
        tag!("solid ")
            >> take_while1!(not_line_ending)
            >> line_ending
            >> triangles: many1!(triangle_ascii)
            >> tag!("endsolid")
            >> opt!(rest)
            >> (build_indexed_mesh(&triangles, 0))
    )
);

named!(
    three_floats<[f32; 3]>,
    do_parse!(f1: ws!(float) >> f2: ws!(float) >> f3: ws!(float) >> ([f1, f2, f3]))
);

named!(vertex, ws!(tag!("vertex")));

named!(
    triangle_ascii<Triangle>,
    do_parse!(
        alt!(ws!(tag!("facet normal")) | tag!("facet normal"))
            >> normal: ws!(three_floats)
            >> ws!(tag!("outer loop"))
            >> vertex
            >> v1: ws!(three_floats)
            >> vertex
            >> v2: ws!(three_floats)
            >> vertex
            >> v3: ws!(three_floats)
            >> ws!(tag!("endloop"))
            >> alt!(ws!(tag!("endfacet")) | tag!("endfacet"))
            >> (Triangle {
                normal: normal,
                vertices: [v1, v2, v3]
            })
    )
);

fn build_indexed_mesh(triangles: &[Triangle], reported_count: u32) -> IndexedMesh {
    let mut indexes = HashMap::new();
    let mut vertices: Vec<Vertex> = Vec::new();

    let indexed_triangles: Vec<IndexedTriangle> = triangles
        .iter()
        .map(|triangle| {
            let mut vertex_indices = [0; 3];

            triangle
                .vertices
                .iter()
                .enumerate()
                .for_each(|(i, vertex)| {
                    let vertex_as_u32_bits =
                        unsafe { std::mem::transmute::<[f32; 3], [u32; 3]>(*vertex) };

                    let index = *indexes
                        .entry(vertex_as_u32_bits)
                        .or_insert_with(|| vertices.len());

                    if index == vertices.len() {
                        vertices.push(*vertex);
                    }

                    vertex_indices[i] = index;
                });

            IndexedTriangle {
                normal: triangle.normal,
                vertices: vertex_indices,
            }
        })
        .collect();

    IndexedMesh {
        reported_triangles_count: reported_count,
        actual_triangles_count: indexed_triangles.len(),
        vertices,
        triangles: indexed_triangles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memmap::MmapOptions;
    use std::io::prelude::*;

    #[test]
    fn parses_both_ascii_and_binary() {
        // derived from: https://www.thingiverse.com/thing:1187833
        let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER.stl").unwrap();
        let moon = unsafe { MmapOptions::new().map(&moon_file).unwrap() };
        let ascii_mesh = parse_stl(&moon);

        assert!(&ascii_mesh.is_ok());

        // credit: https://www.thingiverse.com/thing:26227
        let vase_file = std::fs::File::open("./fixtures/Root_Vase.stl").unwrap();
        let root_vase = unsafe { MmapOptions::new().map(&vase_file).unwrap() };
        let binary_mesh = parse_stl(&root_vase);

        assert!(binary_mesh.is_ok());
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

        assert_eq!(
            triangle,
            Ok((
                vec!().as_slice(),
                Triangle {
                    normal: [0.642777, -0.00000254044, 0.766053],
                    vertices: [
                        [8.08661, 0.373289, 54.1924],
                        [8.02181, 0.689748, 54.2468],
                        [8.10936, 0.0, 54.1733]
                    ]
                }
            ))
        )
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

        let indexed_mesh = parse_stl(mesh_string.as_bytes());

        assert_eq!(
            indexed_mesh,
            Ok((
                vec!().as_slice(),
                IndexedMesh {
                    reported_triangles_count: 0,
                    actual_triangles_count: 2,
                    vertices: vec!(
                        [8.08661, 0.373289, 54.1924],
                        [8.02181, 0.689748, 54.2468],
                        [8.10936, 0.0, 54.1733],
                        [-0.196076, 7.34845, 8.72767],
                        [0.0, 8.11983, 7.87508],
                        [0.0, 7.342, 8.6529]
                    ),
                    triangles: vec!(
                        IndexedTriangle {
                            normal: [0.642777, -0.00000254044, 0.766053],
                            vertices: [0, 1, 2]
                        },
                        IndexedTriangle {
                            normal: [-0.281083, -0.678599, -0.678599],
                            vertices: [3, 4, 5]
                        }
                    )
                }
            ))
        )
    }

    #[test]
    fn does_ascii_from_file() {
        // derived from: https://www.thingiverse.com/thing:1187833
        let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER.stl").unwrap();
        let moon = unsafe { MmapOptions::new().map(&moon_file).unwrap() };
        let mesh = parse_stl(&moon);

        assert!(&mesh.is_ok());
        assert_eq!(&mesh.unwrap().0, b"")
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

        assert_eq!(
            triangle_binary(triangle_bytes),
            Ok((
                vec!().as_slice(),
                Triangle {
                    normal: normal,
                    vertices: [v1, v2, v3]
                }
            ))
        );
    }

    #[test]
    fn parses_mesh() {
        let header = vec![0; 80];
        let count = vec![0; 4];
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

        let all = [header, count, body].concat();

        assert_eq!(
            mesh(&all),
            Ok((
                vec!().as_slice(),
                Mesh {
                    reported_count: 0,
                    triangles: vec!(
                        Triangle {
                            normal: [0.0, 0.0, 0.0],
                            vertices: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                        },
                        Triangle {
                            normal: [0.0, 0.0, 0.0],
                            vertices: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                        },
                    ),
                }
            ))
        );
    }

    #[test]
    fn parses_indexed_mesh() {
        let header = vec![0; 80];
        let count = vec![0; 4];
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

        let all = [header, count, body].concat();

        assert_eq!(
            indexed_mesh_binary(&all),
            Ok((
                vec!().as_slice(),
                IndexedMesh {
                    reported_triangles_count: 0,
                    actual_triangles_count: 2,
                    triangles: vec!(
                        IndexedTriangle {
                            normal: [0.0, 0.0, 0.0],
                            vertices: [0, 0, 0]
                        },
                        IndexedTriangle {
                            normal: [0.0, 0.0, 0.0],
                            vertices: [0, 0, 0]
                        },
                    ),
                    vertices: vec!([0.0, 0.0, 0.0])
                }
            ))
        );
    }

    #[test]
    fn does_binary_from_file() {
        // credit: https://www.thingiverse.com/thing:26227
        let file = std::fs::File::open("./fixtures/Root_Vase.stl").unwrap();
        let root_vase = unsafe { MmapOptions::new().map(&file).unwrap() };
        let start = std::time::Instant::now();
        let mesh = parse_stl(&root_vase);
        let end = std::time::Instant::now();
        println!("root_vase time: {:?}", end - start);

        assert!(mesh.is_ok());
        assert_eq!(mesh.unwrap().0, b"")
    }

    #[test]
    fn does_binary_from_file_starting_with_solid() {
        // credit: https://www.thingiverse.com/thing:26227
        let file = std::fs::File::open("./fixtures/Root_Vase_solid_start.stl").unwrap();
        let root_vase = unsafe { MmapOptions::new().map(&file).unwrap() };
        let mesh = parse_stl(&root_vase);

        assert!(mesh.is_ok());
        assert_eq!(mesh.unwrap().0, b"")
    }

    #[test]
    fn does_ascii_file_without_a_closing_solid_name() {
        // derived from: https://www.thingiverse.com/thing:1187833
        let moon_file =
            std::fs::File::open("./fixtures/MOON_PRISM_POWER_no_closing_name.stl").unwrap();
        let moon = unsafe { MmapOptions::new().map(&moon_file).unwrap() };
        let mesh = parse_stl(&moon);
        let (remaining, result) = mesh.unwrap();
        assert_eq!(remaining, &[]);
        assert_eq!(result.triangles.len(), 3698);
    }

    #[test]
    fn parses_stl_with_dos_line_endings_crlf() {
        // derived from: https://www.thingiverse.com/thing:1187833

        let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER_dos.stl").unwrap();
        let moon = unsafe { MmapOptions::new().map(&moon_file).unwrap() };
        let embedded_res = parse_stl(&moon);
        let (remaining, result) = embedded_res.unwrap();
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

            TestResult::from_bool(parse_stl(&xs).is_ok())
        }

        let mut qc = QuickCheck::new();
        qc.quickcheck(parses_binary_stl_with_at_least_one_triangle as fn(Vec<u8>) -> TestResult);
        qc.min_tests_passed(200);
    }
}
