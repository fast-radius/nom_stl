nom_stl
=======

# What is this

`nom_stl` is a binary and ASCII STL parser, written in Rust, using the [nom](https://github.com/Geal/nom) parser combinator library.
It parses a 30M binary STL in ~250ms.
The [hashbrown](https://github.com/Amanieu/hashbrown) hashmap library is available behind the opt-in `hashbrown` feature flag,
giving a free ~30% speedup, parsing the same file in ~180ms.
`nom_stl` attempts to be mostly API compatible with [stl_io](https://github.com/hmeyer/stl_io), but is a new implementation rather than a fork.

# What does it look like
There isn't really a public API yet, but this is what a test case looks like:

```rust
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
```

# Running the tests faster

To make the tests run faster (but increase build time), you can run the tests in `release` mode.
To do this, run the following:

```
cargo test --release --features=hashbrown -- --nocapture
```

This will also use Hashbrown (a faster hashtable library). Hashbrown is totally optional, but will increase speed.


# What does it need

- [ ] A solid public API
- [ ] Better tests, with better input data rather than 0's for some of the smaller parsers
- [x] Testing around parsing Windows/DOS line-ending files
- [ ] Real documentation/rustdoc
- [ ] A license
- [ ] A home

# Creative Commons

Attribution and thanks goes to the following people for licensing their files Creative Commons so we could include them in this project:

- C4robotics for [Sailor Moon Disguise Pen](https://www.thingiverse.com/thing:1187833)
- virtox for [Binary Roots](https://www.thingiverse.com/thing:26227)
