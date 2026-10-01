N = 10

@testset "2D GF" begin
  # Test 2d getindex setindex!
  data = zeros(ComplexF64, N, N)

  lgf = GreenFunction(copy(data), SkewHermitian)
  ggf = GreenFunction(copy(data), SkewHermitian)

  v = 30 + 30im
  lgf[2, N] = v
  ggf[N, 2] = v

  @test lgf.data[N, 2] == -adjoint(v)
  @test ggf.data[2, N] == -adjoint(v)

  @test lgf[2, N] == -adjoint(lgf.data[N, 2])
  @test ggf.data[2, N] == -adjoint(ggf[N, 2])
end

@testset "Arithmetic preserves the resulting symmetry" begin
  G = GreenFunction(ComplexF64[-im 1+2im; -1+2im -3im], SkewHermitian)
  for H in (imag(G), real(G), conj(G), 2G, G * 2, G / 2, 2 \ G)
    before = copy(H.data)
    H[1, 2] = H[1, 2]
    @test H.data == before
  end
  @test imag(G)[2, 1] == transpose(imag(G)[1, 2])

  for a in (im, 1 + 2im, 2 + 0im)
    for (H, expected) in ((a * G, a * G.data), (G * a, G.data * a),
                          (G / a, G.data / a), (a \ G, a \ G.data))
      @test H isa Matrix
      @test H == expected
    end
  end

  G = GreenFunction(zeros(ComplexF64, 2, 2, 2, 2), SkewHermitian)
  G[1, 2] = [1+2im 3+4im; 5+6im 7+8im]
  H = imag(G)
  before = copy(H.data)
  H[1, 2] = H[1, 2]
  @test H.data == before
  @test H[2, 1] == transpose(H[1, 2])

  for U in (Symmetrical, OnePoint)
    G = GreenFunction(ones(ComplexF64, 2, 2), U)
    @test (im * G).data == im * G.data
  end
end

struct ConjugateTestSymmetry <: KadanoffBaym.AbstractSymmetry end
KadanoffBaym.symmetry(::Type{ConjugateTestSymmetry}) = conj

@testset "Custom involutions do not imply arithmetic closure" begin
  G = GreenFunction(ComplexF64[1 1+2im; 1-2im 2], ConjugateTestSymmetry)
  for f in (x -> -x, conj, real, imag, zero,
            x -> im * x, x -> x * im, x -> x / im, x -> im \ x,
            x -> 2 * x, x -> x * 2, x -> x / 2, x -> 2 \ x)
    H = f(G)
    @test H isa Matrix
    @test H == f(G.data)
  end
end

@testset "Whole-container transpose follows Base array semantics" begin
  scalar = GreenFunction(ComplexF64[-im 1+2im; -1+2im -3im], SkewHermitian)
  blocks = GreenFunction([v * ones(2, 2) for v in scalar.data], SkewHermitian)
  single = GreenFunction(reshape(ComplexF64.(1:6), 2, 3), OnePoint)
  for G in (scalar, blocks, single), f in (transpose, adjoint)
    H = f(G)
    @test !(H isa KadanoffBaym.AbstractGreenFunction)
    @test size(H) == size(f(G.data))
    @test collect(H) == collect(f(G.data))
  end
  G = GreenFunction(zeros(ComplexF64, 2, 2, 2, 2), SkewHermitian)
  @test_throws ErrorException transpose(G)
  @test_throws ErrorException adjoint(G)
end

@testset "Copy and dense resize" begin
  for dims in ((3, 3), (2, 2, 3, 3))
    data = reshape(ComplexF64.(1:prod(dims)), dims)
    G = GreenFunction(copy(data), SkewHermitian)
    H = copy(G)
    @test typeof(H) == typeof(G)
    @test H.data == G.data
    @test H.data !== G.data

    resize!(H, 5)
    @test size(H) == (dims[1:end-2]..., 5, 5)
    @test H.data[ntuple(_ -> Colon(), length(dims)-2)..., 1:3, 1:3] == data
    resize!(H, 2)
    @test H.data == data[ntuple(_ -> Colon(), length(dims)-2)..., 1:2, 1:2]
    @test G.data == data
  end

  for dims in ((3,), (2, 3))
    data = reshape(collect(1:prod(dims)), dims)
    G = GreenFunction(copy(data), OnePoint)
    H = copy(G)
    @test H.data == data
    @test H.data !== G.data
    resize!(H, 5)
    @test size(H) == (dims[1:end-1]..., 5)
    @test H.data[ntuple(_ -> Colon(), length(dims)-1)..., 1:3] == data
    resize!(H, 2)
    @test H.data == data[ntuple(_ -> Colon(), length(dims)-1)..., 1:2]
  end
end

@testset "Packed resize preserves time diagonals" begin
  for v in (1.0im, ComplexF64[im 0; 0 2im])
    G = KadanoffBaym.SkewHermitianArray(v)
    resize!(G, 2)
    G[2, 1] = 3v
    G[2, 2] = 2v
    resize!(G, 4)
    @test size(G) == (4, 4)
    @test G[1, 1] == v
    @test G[2, 1] == 3v
    @test G[1, 2] == -adjoint(3v)
    @test G[2, 2] == 2v
    resize!(G, 1)
    resize!(G, 3)
    @test G[1, 1] == v
    G[3, 1] = 4v
    @test G[1, 3] == -adjoint(4v)
  end
end

@testset "Packed storage is a matrix of values" begin
  for v in (1.0im, ComplexF64[im 0; 0 2im], sparse(ComplexF64[im 0; 0 2im]))
    G = KadanoffBaym.SkewHermitianArray(v)
    resize!(G, 2)
    G[2, 1] = 3v
    G[2, 2] = 2v
    expected = [G[i, j] for i in 1:2, j in 1:2]
    @test ndims(G) == 2
    @test size(G) == (2, 2)
    @test eltype(G) == typeof(v)
    @test all(x -> x isa eltype(G), G)
    @test G[CartesianIndex(2, 1)] == 3v
    @test G[2] == 3v
    @test collect(G) == expected
    @test Array(G) == expected
    @test norm(G) == norm(expected)
    @test collect(adjoint(G)) == -expected
  end
end

@testset "Resizing to the existing shape preserves storage" begin
  for data in (zeros(ComplexF64, 3, 3), zeros(ComplexF64, 2, 2, 3, 3),
               [zeros(ComplexF64, 2, 2) for _ in 1:3, _ in 1:3])
    G = GreenFunction(data, SkewHermitian)
    @test resize!(G, 3) === G
    @test G.data === data
  end
  for data in (zeros(3), zeros(2, 3), [zeros(2, 2) for _ in 1:3])
    G = GreenFunction(data, OnePoint)
    @test resize!(G, 3) === G
    @test G.data === data
  end
  for v in (1.0im, ComplexF64[im 0; 0 2im])
    G = KadanoffBaym.SkewHermitianArray(v)
    resize!(G, 3)
    data = G.data
    diagonals = copy(data)
    @test resize!(G, 3) === G
    @test G.data === data
    @test all(G.data[i] === diagonals[i] for i in eachindex(diagonals))
    @test G[1, 1] === v
  end
end

@testset "4D GF" begin
  # Test 4d getindex setindex!
  data = zeros(ComplexF64, N, N, N, N)

  lgf = GreenFunction(copy(data), SkewHermitian)
  ggf = GreenFunction(copy(data), SkewHermitian)

  v = rand(ComplexF64, N, N)
  lgf[2, N] = v
  ggf[N, 2] = v

  @test lgf.data[:, :, N, 2] == -adjoint(v)
  @test ggf.data[:, :, 2, N] == -adjoint(v)

  @test lgf[2, N] == -adjoint(lgf.data[:, :, N, 2])
  @test ggf.data[:, :, 2, N] == -adjoint(ggf[N, 2])
end

@testset "Mixed indices preserve time-coordinate indexing" begin
  data = reshape(collect(1:36), 3, 3, 2, 2)
  G = GreenFunction(data, SkewHermitian)
  @test G[:, 1, 1] == data[:, :, 1, 1]
  @test G[1, :, 1] == data[:, 1, :, 1]
  @test G[:, 1] == data[:, :, :, 1]
  @test G[1, Int32(1)] == data[:, :, 1, 1]
  @test G[1:2, 1] == data[:, :, 1:2, 1]
  @test G[:, :, 1, 1] == data[:, :, 1, 1]

  data = reshape(collect(1:12), 2, 2, 3)
  G = GreenFunction(copy(data), OnePoint)
  @test G[:, 2] == data[:, :, 2]
  @test G[1, Int32(2)] == data[:, 1, 2]
  G[:, 2] = fill(7, 2, 2)
  @test G[2] == fill(7, 2, 2)
  G[1, Int32(3)] = [8, 9]
  @test G.data[:, 1, 3] == [8, 9]
end

@testset "Cartesian reads obey the AbstractArray interface" begin
  for (data, U) in (([1.0 2; 2 3], Symmetrical),
                    (ComplexF64[-im 1+2im; -1+2im -3im], SkewHermitian),
                    (reshape(ComplexF64.(1:36), 2, 2, 3, 3), SkewHermitian),
                    ([1.0, 2, 3], OnePoint),
                    (reshape(collect(1.0:12.0), 2, 2, 3), OnePoint))
    G = GreenFunction(data, U)
    @test [G[I] for I in CartesianIndices(G)] == data
    @test Array(G) == data
    @test collect(G) == data
    @test sin.(G) == sin.(data)
    @test sum(G) == sum(data)
    @test norm(G) ≈ norm(data)
    @test G + G == data + data
  end
  G = GreenFunction(zeros(2, 2, 3, 3), SkewHermitian)
  @test_throws ErrorException G[1]
  @test size(G[3, 2]) == (2, 2)
  J = GreenFunction(zeros(2, 2, 3), OnePoint)
  @test size(J[3]) == (2, 2)
end

@testset "Views retain their underlying array axes" begin
  I = Base.IdentityUnitRange(2:3)
  for (data, U) in ((view(reshape(collect(1:16), 4, 4), I, I), Symmetrical),
                    (view(reshape(collect(1:256), 4, 4, 4, 4), I, I, I, I), SkewHermitian),
                    (view(collect(1:4), I), OnePoint))
    G = GreenFunction(data, U)
    @test axes(G) == axes(data)
    @test all(axes(G, d) == axes(data, d) for d in 1:ndims(data))
    @test CartesianIndices(G) == CartesianIndices(data)
    @test all(G[J] == data[J] for J in CartesianIndices(data))
    @test collect(G) == collect(data)
  end
end

@testset "Matrix assignments preserve values and ownership" begin
  data = reshape(complex.(1:16, 16:-1:1), 2, 2, 2, 2)
  for (i, j) in ((1, 2), (2, 1), (1, 1))
    G = GreenFunction(copy(data), SkewHermitian)
    v = copy(data[:, :, i, j])
    G[1, 2] = view(G.data, :, :, i, j)
    @test G[1, 2] == v
    @test G[2, 1] == -adjoint(v)
  end

  G = GreenFunction(zeros(ComplexF64, 2, 2, 2, 2), SkewHermitian)
  v = ComplexF64[1+2im 3+4im; 5+6im 7+8im]
  G[1, 2] = v
  v[1, 1] = 0
  @test G[1, 2][1, 1] == 1+2im
  @test G[2, 1] == -adjoint(G[1, 2])

  G = GreenFunction([zeros(ComplexF64, 2, 2) for _ in 1:2, _ in 1:2], SkewHermitian)
  G[1, 2] = v
  @test G[1, 2] === v
  @test G[2, 1] == -adjoint(v)
end

@testset "Matrix assignments retain Julia's shape checks" begin
  for shape in ((1, 1), (4, 1), (2, 3))
    G = GreenFunction(zeros(ComplexF64, 2, 2, 3, 3), SkewHermitian)
    @test_throws DimensionMismatch setindex!(G, ones(ComplexF64, shape), 1, 2)
    @test iszero(G.data)
  end
  G = GreenFunction(zeros(ComplexF64, 2, 2, 3, 3), SkewHermitian)
  v = reshape(ComplexF64.(1:4), 1, 4)
  G[1, 1] = v
  @test G[1, 1] == reshape(vec(v), 2, 2)
  @test_throws BoundsError setindex!(G, ones(2, 2), 1, 4)
  rectangular = GreenFunction(zeros(2, 3, 2, 2), SkewHermitian)
  @test_throws DimensionMismatch setindex!(rectangular, ones(2, 3), 1, 2)
end

@testset "Mirroring uses the stored block shape and precision" begin
  values = complex.([1.1, 2.2, 3.3, 4.4], [0.2, 0.4, 0.6, 0.8])
  for U in (Symmetrical, SkewHermitian), T in (ComplexF32, ComplexF64),
      v in (values, reshape(values, 1, 4)), (i, j) in ((1, 2), (2, 1))
    G = GreenFunction(zeros(T, 2, 2, 2, 2), U)
    G[i, j] = v
    expected = reshape(T.(values), 2, 2)
    @test G[i, j] == expected
    @test G[j, i] == KadanoffBaym.symmetry(U)(expected)
  end
end

@testset "Transformed views preserve the assigned symmetry" begin
  data = reshape(complex.(1:16, 16:-1:1), 2, 2, 2, 2)
  for U in (Symmetrical, SkewHermitian), source in ((1, 2), (2, 1)), transform in (transpose, adjoint, x -> view(x, 2:-1:1, :))
    G = GreenFunction(copy(data), U)
    v = transform(view(G.data, :, :, source...))
    expected = copy(v)
    G[1, 2] = v
    @test G[1, 2] == expected
    @test G[2, 1] == KadanoffBaym.symmetry(U)(expected)
  end
end

@testset "Base functions & setindex!" begin
  # Test AbstractArray-like behaviour
  data = rand(ComplexF64, N, N, N, N)
  gf = GreenFunction(copy(data), SkewHermitian)

  @test (-gf).data == (-data)
  @test (conj(gf)).data == conj(data)
  @test (real(gf)).data == real(data)
  @test (imag(gf)).data == imag(data)
end

@testset "setindex!" begin
  data = zeros(ComplexF64, N, N, N, N)
  gf = GreenFunction(copy(data), SkewHermitian)

  b = rand(ComplexF64, N, N)

  gf[1, 2] = b
  data[:, :, 1, 2] = b
  data[:, :, 2, 1] = -adjoint(b)
  @test gf.data == data

  # # this used to fail
  # gf[:, :, 2, 1] = b
  # data[:, :, 2, 1] = b
  # data[:, :, 1, 2] = -adjoint(b)

  for i in 1:N, j in 1:i
    data[:, :, i, j] = b
    if j != i
      data[:, :, j, i] = -adjoint(b)
    end
  end
  
  for i in 1:N, j in 1:i
    gf[i, j] = b
  end

  @test gf.data == data
end
