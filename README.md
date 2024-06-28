# Heapless Matrix Library Crate

This crate provides a simple and efficient implementation of a matrix data structure in Rust. It offers functionality for creating, manipulating, and performing operations on matrices, such as transposition, summation, and multiplication.

## Features

- **Generic Matrix Type**: Define matrices of any size with fixed dimensions using const generics.
- **Element-wise Operations**: Perform operations such as transposition, summation, and multiplication on matrices.
- **Compile-time Safety**: Ensure that matrix dimensions are enforced at compile-time, preventing runtime errors.
- **Efficient Memory Usage**: Utilizes the `heapless` crate for fixed-capacity vectors, ensuring efficient memory usage and avoiding heap allocations.
- **Suitable for Bare Metal Coding**: This implementation does not require heap allocation, making it suitable for bare metal programming and environments with limited resources.
- **Clear API**: Provides a clear and concise API for working with matrices, making it easy to integrate into your Rust projects.

## Usage

To use this crate in your Rust project, add it as a dependency in your `Cargo.toml` file:

```toml
[dependencies]
heapless_matrix = {git = "https://github.com/Spago123/heapless-matrix"}
```

## Note
From version `0.1.4` the crate has implemented the functionalities to be used in a `no_std` environment.

## Contributing
We welcome contributions to the Heapless Matrix Library Crate! The goal is to add more features that are connected to matrices and data - analisys such as least - squares, optimization algorithms, singular - value decomposition (SVD), control - algorithms etc... Everything in the context of the heapless - crate and application on bare metal and real - time.

## Contact
If you have any suggestions how to improve the crate or you encounter possible bugs or have questions feel free to open an issue in the GitHub repository.
