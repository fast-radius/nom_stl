#[macro_use]
extern crate criterion;

use criterion::Criterion;
use memmap::MmapOptions;
use nom_stl::parse_stl;

fn parse_stl_binary(c: &mut Criterion) {
    let vase_file = std::fs::File::open("./fixtures/Root_Vase.stl").unwrap();
    let root_vase = unsafe { MmapOptions::new().map(&vase_file).unwrap() };

    c.bench_function("parse_stl_root_vase", move |b| {
        b.iter(|| parse_stl(&root_vase))
    });
}

fn parse_stl_ascii(c: &mut Criterion) {
    let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER.stl").unwrap();
    let moon = unsafe { MmapOptions::new().map(&moon_file).unwrap() };

    c.bench_function("parse_stl_moon_prism_power", move |b| {
        b.iter(|| parse_stl(&moon))
    });
}

criterion_group!(benches, parse_stl_binary, parse_stl_ascii);
criterion_main!(benches);
