using ERGMRank
using Graphs
using Test

@testset "ERGMRank.jl" begin
    @testset "Module loading" begin
        @test @isdefined(ERGMRank)
    end

    @testset "RankNetwork construction" begin
        rnet = RankNetwork(5)
        @test rnet isa RankNetwork{Int}
        @test Graphs.nv(rnet) == 5
        @test Graphs.ne(rnet) == 0

        rnet2 = RankNetwork(4; max_rank=3)
        @test rnet2.max_rank == 3
    end

    @testset "RankNetwork operations" begin
        rnet = RankNetwork(4; max_rank=3)
        ERGMRank.set_rank!(rnet, 1, 2, 1)
        @test ERGMRank.get_rank(rnet, 1, 2) == 1
        @test ERGMRank.get_rank(rnet, 2, 1) === nothing

        ERGMRank.set_rank!(rnet, 1, 3, 2)
        rankings = ERGMRank.get_rankings_by(rnet, 1)
        @test length(rankings) == 2

        @test_throws ArgumentError ERGMRank.set_rank!(rnet, 1, 1, 1)  # self-rank
        @test_throws ArgumentError ERGMRank.set_rank!(rnet, 1, 2, 4)  # rank > max_rank
    end

    @testset "Reference measures" begin
        @test PlackettLuce() isa PlackettLuce
        @test ThurstoneMosteller() isa ThurstoneMosteller
        @test ThurstoneMosteller(2.0) isa ThurstoneMosteller
    end

    @testset "Rank ERGM terms" begin
        @test RankEdges() isa RankEdges
        @test RankMutual() isa RankMutual
        @test RankMutual(3) isa RankMutual
        @test RankTransitivity() isa RankTransitivity
        @test RankNonconsensus() isa RankNonconsensus
        @test RankDeference() isa RankDeference
        @test RankLocaltriangle() isa RankLocaltriangle
        @test RankNodecov(:age) isa RankNodecov
        @test RankAbsdiff(:age) isa RankAbsdiff
    end

    @testset "Rank matrix conversion" begin
        rnet = RankNetwork(3; max_rank=2)
        ERGMRank.set_rank!(rnet, 1, 2, 1)
        ERGMRank.set_rank!(rnet, 1, 3, 2)
        mat = rank_matrix(rnet)
        @test mat[1, 2] == 1
        @test mat[1, 3] == 2
        @test ismissing(mat[2, 1])
    end

    @testset "Estimation API" begin
        @test ergm_rank === fit_rank_ergm
    end

    @testset "Simulation" begin
        @test isdefined(ERGMRank, :simulate_rank_ergm)
    end
end
