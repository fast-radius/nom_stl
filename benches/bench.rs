#[macro_use]
extern crate criterion;

use criterion::Criterion;
use memmap::MmapOptions;
use nom_stl::parse_stl;

fn parse_stl_binary(c: &mut Criterion) {
    let moon_file = std::fs::File::open("./fixtures/MOON_PRISM_POWER_binary.stl").unwrap();
    let moon = unsafe { MmapOptions::new().map(&moon_file).unwrap() };

    c.bench_function("parse_stl_moon_prism_power_binary", move |b| {
        b.iter(|| parse_stl(&moon))
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
