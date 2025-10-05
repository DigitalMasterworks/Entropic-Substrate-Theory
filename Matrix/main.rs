use ndarray::Array1;
use ndarray_npy::NpzWriter;
use std::{env, error::Error, fs::File};

fn idx(x: usize, y: usize, nx: usize) -> usize { y * nx + x }

/// Build 2D 5-point Laplacian (Dirichlet) on an NX×NY grid, row-major flattening.
/// Column order per row is kept sorted: up, left, center, right, down.
fn build_laplacian2d_csr(nx: usize, ny: usize) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let n = nx * ny;
    let mut data = Vec::with_capacity(5 * n);
    let mut indices = Vec::with_capacity(5 * n);
    let mut indptr = Vec::with_capacity(n + 1);

    let mut nnz: i32 = 0;
    indptr.push(0);

    for y in 0..ny {
        for x in 0..nx {
            let row = idx(x, y, nx);

            // up (row - nx)
            if y > 0 {
                data.push(1.0);
                indices.push((row - nx) as i32);
                nnz += 1;
            }
            // left (row - 1)
            if x > 0 {
                data.push(1.0);
                indices.push((row - 1) as i32);
                nnz += 1;
            }
            // center
            data.push(-4.0);
            indices.push(row as i32);
            nnz += 1;

            // right (row + 1)
            if x + 1 < nx {
                data.push(1.0);
                indices.push((row + 1) as i32);
                nnz += 1;
            }
            // down (row + nx)
            if y + 1 < ny {
                data.push(1.0);
                indices.push((row + nx) as i32);
                nnz += 1;
            }
            indptr.push(nnz);
        }
    }
    (data, indices, indptr)
}

/// Replicate a CSR block along the diagonal CF times (no cross-component coupling).
fn block_diag_repeat(
    data: &[f64],
    indices: &[i32],
    indptr: &[i32],
    n: usize,
    cf: usize,
) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    if cf == 1 {
        return (data.to_vec(), indices.to_vec(), indptr.to_vec());
    }
    let per_nnz = *indptr.last().unwrap() as i32;

    let mut out_data = Vec::with_capacity(data.len() * cf);
    let mut out_indices = Vec::with_capacity(indices.len() * cf);
    let mut out_indptr: Vec<i32> = Vec::with_capacity(cf * n + 1);
    out_indptr.push(0);

    let mut current_nnz: i32 = 0;
    for c in 0..cf {
        // copy data
        out_data.extend_from_slice(data);
        // shift indices by component offset
        let offset = (c * n) as i32;
        out_indices.extend(indices.iter().map(|&j| j + offset));
        // extend indptr for this block
        for &v in indptr.iter().skip(1) {
            out_indptr.push(current_nnz + v);
        }
        current_nnz += per_nnz;
    }
    (out_data, out_indices, out_indptr)
}

/// Save SciPy-compatible CSR .npz with keys: data (f64), indices (i32), indptr (i32), shape (i64[2]).
fn save_csr_npz(
    path: &str,
    data: &[f64],
    indices: &[i32],
    indptr: &[i32],
    nrows: usize,
    ncols: usize,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut npz = NpzWriter::new(file);

    npz.add_array("data", &Array1::from_vec(data.to_vec()))?;
    npz.add_array("indices", &Array1::from_vec(indices.to_vec()))?;
    npz.add_array("indptr", &Array1::from_vec(indptr.to_vec()))?;
    npz.add_array("shape", &Array1::from_vec(vec![nrows as i64, ncols as i64]))?;
    npz.finish()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Args: NX NY [CF] [OUT]
    let mut args = env::args().skip(1);
    let nx: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let ny: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);
    let cf: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let out = args
        .next()
        .unwrap_or_else(|| format!("Laplacian2D_{}x{}_Cf{}.npz", nx, ny, cf));

    let n = nx * ny;
    let (d0, i0, p0) = build_laplacian2d_csr(nx, ny);
    let (data, indices, indptr) = block_diag_repeat(&d0, &i0, &p0, n, cf);
    let dim = cf * n;

    println!(
        "grid: {}x{}, components: {}, matrix: {}x{}, nnz: {}",
        nx,
        ny,
        cf,
        dim,
        dim,
        data.len()
    );

    save_csr_npz(&out, &data, &indices, &indptr, dim, dim)?;
    println!("{}", out);

    // Tiny 8x8 corner preview
    let b = 8usize.min(dim);
    println!("\n--- {}x{} Corner Block ---", b, b);
    for r in 0..b {
        let start = indptr[r] as usize;
        let end = indptr[r + 1] as usize;
        let mut row = vec![0.0f64; b];
        for p in start..end {
            let j = indices[p] as usize;
            if j < b {
                row[j] = data[p];
            }
        }
        println!(
            "{}",
            row.iter()
                .map(|v| format!("{:>3.0}", v))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    Ok(())
}

