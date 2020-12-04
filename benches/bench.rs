#[macro_use]
extern crate criterion;

use criterion::Criterion;
use nom_stl::parse_stl;
use std::fs::File;
use std::io::{BufReader, Read};

fn parse_stl_binary_big(c: &mut Criterion) {
    let root_vase_file = File::open("./fixtures/Root_Vase.stl").unwrap();
    let mut root_vase = BufReader::new(&root_vase_file);
    let mut root_vase_buf = vec![];
    root_vase.read_to_end(&mut root_vase_buf).unwrap();

    let mut group = c.benchmark_group("big");

    group.sample_size(15);

    group.bench_function("parse_stl_root_vase_binary_big_unindexed", move |b| {
        b.iter(|| parse_stl(&mut root_vase_buf).unwrap())
    });

    group.finish();
}

fn parse_stl_binary(c: &mut Criterion) {
    let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER_binary.stl").unwrap();
    let mut moon = BufReader::new(&moon_file);
    let mut moon_buf = vec![];
    moon.read_to_end(&mut moon_buf).unwrap();

    c.bench_function("parse_stl_moon_prism_power_binary", move |b| {
        b.iter(|| parse_stl(&mut moon_buf).unwrap())
    });
}

fn parse_stl_ascii(c: &mut Criterion) {
    let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER.stl").unwrap();
    let mut moon = BufReader::new(&moon_file);
    let mut moon_buf = vec![];
    moon.read_to_end(&mut moon_buf).unwrap();

    c.bench_function("parse_stl_moon_prism_power_ascii", move |b| {
        b.iter(|| parse_stl(&mut moon_buf).unwrap())
    });
}

criterion_group!(
    benches,
    parse_stl_binary_big,
    parse_stl_binary,
    parse_stl_ascii
);
criterion_main!(benches);
