use nom::*;

#[cfg(feature = "hashbrown")]
use hashbrown::HashMap;
#[cfg(not(feature = "hashbrown"))]
use std::collections::HashMap;

type Vertex = [f32; 3];
type Index = usize;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triangle {
    normal: Vertex,
    vertices: [Vertex; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    reported_count: u32,
    triangles: Vec<Triangle>,
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

// BINARY
/////////////////////////////////////////////////////////////////

named!(
    triangle_binary<Triangle>,
    do_parse!(
        normal: count_fixed!(f32, le_f32, 3)
            >> v1: count_fixed!(f32, le_f32, 3)
            >> v2: count_fixed!(f32, le_f32, 3)
            >> v3: count_fixed!(f32, le_f32, 3)
            >> _attribute_byte_count: take!(2)
            >> (Triangle {
                normal: normal,
                vertices: [v1, v2, v3]
            })
    )
);

named!(
    mesh<Mesh>,
    do_parse!(
        _header: complete!(take!(80))
            >> reported_count: complete!(le_u32)
            >> triangles: many1!(complete!(triangle_binary))
            >> (Mesh {
                reported_count: reported_count,
                triangles: triangles
            })
    )
);

named!(
    pub indexed_mesh_binary<IndexedMesh>,
    do_parse!(
        _not_solid: not!(tag!("solid")) >>
            _header: complete!(take!(80))
            >> reported_count: complete!(le_u32)
            >> triangles: many0!(complete!(triangle_binary))
            >> (build_indexed_mesh(&triangles, reported_count))
    )
);

// ASCII
/////////////////////////////////////////////////////////////////

named!(
    triangle_ascii<Triangle>,
    do_parse!(
        alt!(ws!(tag!("facet normal")) | tag!("facet normal"))
            >> normal: ws!(count_fixed!(f32, ws!(float), 3))
            >> ws!(tag!("outer loop"))
            >> ws!(tag!("vertex"))
            >> v1: ws!(count_fixed!(f32, ws!(float), 3))
            >> ws!(tag!("vertex"))
            >> v2: ws!(count_fixed!(f32, ws!(float), 3))
            >> ws!(tag!("vertex"))
            >> v3: ws!(count_fixed!(f32, ws!(float), 3))
            >> ws!(tag!("endloop"))
            >> tag!("endfacet")
            >> (Triangle {
                normal: normal,
                vertices: [v1, v2, v3]
            })
    )
);

named!(
    pub indexed_mesh_ascii<IndexedMesh>,
    do_parse!(
        tag!("solid ")
            >> many1!(not_line_ending)
            >> newline
            >> triangles: many1!(complete!(triangle_ascii))
            >> ws!(tag!("endsolid"))
            >> rest
            >> (build_indexed_mesh(&triangles, 0))
    )
);

fn build_indexed_mesh(triangles: &[Triangle], reported_count: u32) -> IndexedMesh {
    let mut indexes = HashMap::new();
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indexed_triangles: Vec<IndexedTriangle> = Vec::with_capacity(triangles.len());

    for triangle in triangles {
        let mut vertex_indices = [0; 3];

        for (i, vertex) in triangle.vertices.iter().enumerate() {
            let vertex_as_u32_bits = unsafe { std::mem::transmute::<[f32; 3], [u32; 3]>(*vertex) };

            let index = *indexes
                .entry(vertex_as_u32_bits)
                .or_insert_with(|| vertices.len());

            if index == vertices.len() {
                vertices.push(*vertex);
            }

            vertex_indices[i] = index;
        }

        indexed_triangles.push(IndexedTriangle {
            normal: triangle.normal,
            vertices: vertex_indices,
        });
    }

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
    use std::io::prelude::*;

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

        let indexed_mesh = indexed_mesh_ascii(mesh_string.as_bytes());

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
    fn does_big_one_ascii() {
        // let start = std::time::Instant::now();
        let mut moon =
            // std::fs::File::open("/Users/clark/code/stl_nom/MOON_PRISM_POWER.stl").unwrap();
            std::fs::File::open("/Users/clark/code/data-misc/partcomplex.app/jrs.stl").unwrap();

        let mut buf = Vec::new();
        moon.read_to_end(&mut buf).unwrap();
        let mesh = indexed_mesh_ascii(&buf);
        // let end = std::time::Instant::now();

        // println!("{:?}", end - start);

        assert!(&mesh.is_ok());
    }

    #[test]
    fn parses_triangles() {
        assert_eq!(
            triangle_binary(&[
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // normal
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // v1
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // v2
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // v3
                0, 0 // uint16
            ]),
            Ok((
                vec!().as_slice(),
                Triangle {
                    normal: [0.0, 0.0, 0.0],
                    vertices: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
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
    fn binary_does_not_parse_ascii() {
        // this file is an ascii stl
        // credit: https://www.thingiverse.com/thing:1187833
        let mut sailor_moon =
            std::fs::File::open("/Users/clark/code/stl_nom/MOON_PRISM_POWER.stl").unwrap();
        let mut buf = Vec::new();
        sailor_moon.read_to_end(&mut buf).unwrap();
        let mesh = indexed_mesh_binary(&buf);

        assert!(mesh.is_err())
    }

    #[test]
    fn does_big_one() {
        let start = std::time::Instant::now();

        let mut oakley =
        std::fs::File::open("/Users/clark/code/dummy.stl").unwrap();

        let mut buf = Vec::new();
        oakley.read_to_end(&mut buf).unwrap();
        let mesh = indexed_mesh_binary(&buf);
        let end = std::time::Instant::now();

        println!("{:?}", end - start);

        assert!(mesh.is_ok());
    }
}
