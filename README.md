# nom_stl

[![CircleCI](https://circleci.com/gh/fast-radius/nom_stl/tree/master.svg?style=svg&circle-token=3f57317aeed67f5d7eb5a23c0c587bfd98f5bb0b)](https://circleci.com/gh/fast-radius/nom_stl/tree/master)

# What

`nom_stl` is a binary and ASCII STL parser, written in pure Rust, with only one runtime dependency: the [nom](https://github.com/Geal/nom) parser combinator library.
`nom_stl` automatically differentiates between ASCII and binary STLs.
It parses a 30M binary STL in <20ms.

# Use

```rust
let file = std::fs::File::open("./fixtures/Root_Vase.stl").unwrap();
let mut root_vase = BufReader::new(&file);
let mesh: Mesh = parse_stl(&mut root_vase)?;
assert_eq!(mesh.triangles.len(), 596_736);
```

`nom_stl` accepts STL bytes in a wide variety of argument formats: it will try to parse any collection of bytes that implements [Read](https://doc.rust-lang.org/std/io/trait.Read.html) and [Seek](https://doc.rust-lang.org/std/io/trait.Seek.html).

# Running the tests

```
$ cargo test
```

To make the tests run faster (but increase build time), you can run the tests in `release` mode.
To do this, run the following:

```
$ cargo test --release
```

# Running the benchmarks

```
$ cargo bench
```

# What does it need

- [x] A solid public API
- [x] Better tests, with better input data rather than 0's for some of the smaller parsers
- [x] Testing around parsing Windows/DOS line-ending files
- [x] Property testing (https://crates.io/crates/quickcheck)
- [x] Latest Nom (5.1)
- [x] Buffered IO
- [x] Generic normal and vertex types
- [x] Real documentation/rustdoc
- [x] A license
- [x] A home

# Creative Commons

Attribution and thanks goes to the following people for licensing their files Creative Commons so we could include them in this project:

- C4robotics for [Sailor Moon Disguise Pen](https://www.thingiverse.com/thing:1187833)
- virtox for [Binary Roots](https://www.thingiverse.com/thing:26227)
