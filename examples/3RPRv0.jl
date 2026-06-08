# This is case 0 of Example 5.2 in the overleaf file
# Reference: https://arxiv.org/pdf/1903.06126

using Plots, ImplicitPlots, Random, LinearAlgebra, ProjectedHypersurfaceRegions
mkpath("./results/3RPRv0")

Random.seed!(12345)

time_start_round1 = time()

# Incidence variety of the discriminant
a2 = 14 // 10;
a3 = 7 // 10;
b3 = 1;
A2 = 16 // 10;
A3 = 9 // 10;
B3 = 6 // 10;
c3 = 1;

@var p[1:2] φ[1:2] c[1:2]

f = [φ[1]^2 + φ[2]^2 - 1,
    p[1]^2 + p[2]^2 - 2 * (a3 * p[1] + b3 * p[2]) * φ[1] + 2 * (b3 * p[1] - a3 * p[2]) * φ[2] + a3^2 + b3^2 - c[1],
    p[1]^2 + p[2]^2 - 2 * A2 * p[1] + 2 * ((a2 - a3) * p[1] - b3 * p[2] + A2 * a3 - A2 * a2) * φ[1] + 2 * (b3 * p[1] + (a2 - a3) * p[2] - A2 * b3) * φ[2]
    + (a2 - a3)^2 + b3^2 + A2^2 - c[2],
    p[1]^2 + p[2]^2 - 2 * (A3 * p[1] + B3 * p[2]) + A3^2 + B3^2 - c3
]

Jac = differentiate(f, [p; φ])
F = System([f; det(Jac)], variables=[p; φ; c]) # use System(prod(c)*[f;det(Jac)]) to impose c>0

# Form projected hypersurface
h = ProjectedHypersurface(F, c)

# Degree of the discriminant
d = degree(h)
println("Degree of discriminant: $d")

# Set up routing function
center = [4.723952824712583, 4.334474432194565]
# center = 5 * rand(length(c))
write_parameters("./results/3RPRv0/center.txt", center)
r = RoutingFunction(h; c=center);

# Routing points
# pts = read_solutions("./results/3RPRv0/routing_points.txt") |> real
routing_result = critical_points(r)
pts = routing_points(routing_result)
res = trace_result(routing_result)
mon_res = monodromy_result(routing_result)

# Connected components 
partition_result = partition_of_critical_points(r, routing_result)
G = partitions(partition_result)
idx = morse_indices(partition_result)
failures = failed_info(partition_result)

time_end_round1 = time()
println("Computation time for round 1: $(time_end_round1 - time_start_round1) seconds")

h_symbolic(x, y) = 196295728950123786926269531250000 * x^12 - 767203993922710418701171875000000 * x^11 * y + 2089118485600296401977539062500000 * x^10 * y^2 - 4216792583950911712646484375000000 * x^9 * y^3 + 7229376568438361442565917968750000 * x^8 * y^4 - 10206563580314163024902343750000000 * x^7 * y^5 + 12543568168551762908935546875000000 * x^6 * y^6 - 12942052679502311462402343750000000 * x^5 * y^7 + 11621630905397346794128417968750000 * x^4 * y^8 - 8568668984821689056396484375000000 * x^3 * y^9 + 5266387514426347183227539062500000 * x^2 * y^10 - 2264704714452214324951171875000000 * x * y^11 + 704658336671761482238769531250000 * y^12 - 5310424235636101455688476562500000 * x^11 + 10256186900031387405395507812500000 * x^10 * y - 12823816593291740707397460937500000 * x^9 * y^2 + 14314751156360338737182617187500000 * x^8 * y^3 - 15819218037191440601806640625000000 * x^7 * y^4 + 30741949846628109532470703125000000 * x^6 * y^5 - 23371332321325290721435546875000000 * x^5 * y^6 + 13553165556148326097412109375000000 * x^4 * y^7 + 17334257065501188356323242187500000 * x^3 * y^8 - 40633919316425637767944335937500000 * x^2 * y^9 + 34515742520404425130004882812500000 * x * y^10 - 17995045839513484004516601562500000 * y^11 + 66593219792630141888275146484375000 * x^10 - 46309389568053371785949707031250000 * x^9 * y + 7844722817899843661932373046875000 * x^8 * y^2 - 203836715728006536223388671875000000 * x^7 * y^3 + 388918622282993466970690917968750000 * x^6 * y^4 - 592306241692661851811791992187500000 * x^5 * y^5 + 273491489232514302364050292968750000 * x^4 * y^6 - 21108927090811038612451171875000000 * x^3 * y^7 + 82524985815888338933221435546875000 * x^2 * y^8 - 245704207204491902141418457031250000 * x * y^9 + 208485646387667185206829833984375000 * y^10 - 491361663120683120599112548828125000 * x^9 - 103937218167076174065651611328125000 * x^8 * y + 594724196252142059791016601562500000 * x^7 * y^2 + 913911875429192857587493164062500000 * x^6 * y^3 - 738905630650415498301921386718750000 * x^5 * y^4 - 344093032271954459432491699218750000 * x^4 * y^5 - 1205990397851139754254131835937500000 * x^3 * y^6 + 1053347332596399643618485351562500000 * x^2 * y^7 + 988590105062845339208649169921875000 * x * y^8 - 1384687462999934603047335205078125000 * y^9 + 2322745235076766354594665343017578125 * x^8 + 1714579889508108555914827148437500000 * x^7 * y - 3262282958242620314953997934570312500 * x^6 * y^2 - 5832169045971404374965716367187500000 * x^5 * y^3 + 10893356024602613597095853854980468750 * x^4 * y^4 + 5182467095609283379816718789062500000 * x^3 * y^5 - 6479150580989009902595280278320312500 * x^2 * y^6 - 2877262055592290743161334570312500000 * x * y^7 + 5668550991746913206995499014892578125 * y^8 - 7142265829583021050270393216406250000 * x^7 - 7086086901232146261121612175000000000 * x^6 * y + 12013673855930761205874066173437500000 * x^5 * y^2 - 344773183651406946332332113281250000 * x^4 * y^3 - 29248875795995767849668473510156250000 * x^3 * y^4 + 14887889066426959582809434564062500000 * x^2 * y^5 + 6994862132513215429247533453125000000 * x * y^6 - 14454531495390563330263095775781250000 * y^7 + 14000128356458489494295860602039062500 * x^6 + 12813990331583462436936633371062500000 * x^5 * y - 13576603440873500956259078666539062500 * x^4 * y^2 + 18214871420641102794340157441625000000 * x^3 * y^3 + 2970094953908399066555844535835937500 * x^2 * y^4 - 11747557932204677952872867231437500000 * x * y^5 + 22441862644402140088658666550914062500 * y^6 - 16825935213921931226292649863431250000 * x^5 - 11265427987913760671505680885073750000 * x^4 * y + 2837705435814820941159751413375000000 * x^3 * y^2 - 11631832344100905946539992983295000000 * x^2 * y^3 + 2857289652163231862326265170181250000 * x * y^4 - 21000109230453470099996161963106250000 * y^5 + 11756533913278670472222020242791043750 * x^4 + 5334980498995382286764175962992500000 * x^3 * y + 6146267358612943780958443608247962500 * x^2 * y^2 + 8720311962440393560109016264303300000 * x * y^3 + 13704285242854176132877542310369643750 * y^4 - 4588091537074638216251858108952126000 * x^3 - 3984350042378533452211361744684276000 * x^2 * y - 9400791118135269858076343056496508000 * x * y^2 - 8350808578637710149381913639957382000 * y^3 + 1405502878197114916643789827896060220 * x^2 + 4149982062469570296773760547010043120 * x * y + 4605205352798696722790403930897017980 * y^2 - 718739734275658526731075299628542216 * x - 1570152069436761537230300494645675208 * y + 220033696823508718672600054127679749

root_counting_system = System(f, variables=vcat(p, φ), parameters=c);

function analyze_and_save_result()
    write_parameters("./results/3RPRv0/monodromy_parameters.txt", parameters(mon_res))
    write_solutions("./results/3RPRv0/monodromy_result.txt", solutions(mon_res))
    write_solutions("./results/3RPRv0/result.txt", solutions(res))
    write_solutions("./results/3RPRv0/routing_points.txt", pts)
    write("./results/3RPRv0/connected_components.txt", string(G))

    println("Connected components: $(G)")
    println("Indicies: $(idx)")
    println("Failed info: $(failures)")
    println()

    println("Connected components: $(G)")
    println("Indicies: $(idx)")
    println("Failed info: $(failures)")
    println()

    generate_plot(r, routing_result, partition_result;
        h=h_symbolic,
        xlims=(-10, 12),
        ylims=(-10, 12)
    )

    savefig("./figures/3RPR.pdf")
    savefig("./figures/3RPR.svg")
    savefig("./figures/3RPR.png")

    generate_plot(r, routing_result, partition_result;
        h=h_symbolic,
        root_counting_system=root_counting_system,
        markersize=5,
        annotation_textsize=4,
        xlims=(-10, 12),
        ylims=(-10, 12)
    )

    plot!(; xlims=(0, 4.5), ylims=(0, 3.5), legend=false)
    savefig("./figures/3RPR_zoomed_in.pdf")
    savefig("./figures/3RPR_zoomed_in.svg")
    savefig("./figures/3RPR_zoomed_in.png")
end
analyze_and_save_result()

# Try another round of monodromy (only if you think the first attempt missed solutions)
println("Running second round of monodromy...")

time_start_round2 = time()
old_number_of_monodromy_solutions = length(solutions(mon_res))

options = MonodromyOptions(
    parameter_sampler=p -> 100 .* randn(ComplexF64, length(p)), # larger loops
    max_loops_no_progress=10 # stopping criterion
)
routing_result = critical_points(r, solutions(mon_res), parameters(mon_res), options=options)
pts = routing_points(routing_result)
res = trace_result(routing_result)
mon_res = monodromy_result(routing_result)

partition_result = partition_of_critical_points(r, routing_result)
G = partitions(partition_result)
idx = morse_indices(partition_result)
failures = failed_info(partition_result)

time_end_round2 = time()
println("Additional computation time for round 2: $(time_end_round2 - time_start_round2) seconds")
println("Total computation time for round 1 and 2: $((time_end_round1 - time_start_round1) + (time_end_round2 - time_start_round2)) seconds")

if length(solutions(mon_res)) > old_number_of_monodromy_solutions
    println("Found new solutions with additional monodromy round!")
    analyze_and_save_result()
else
    println("No new solutions found in the additional monodromy round.")
end
