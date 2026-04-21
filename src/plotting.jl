export generate_plot

using Plots, ImplicitPlots

function _default_contour_function(r::RoutingFunction, h::Function)
    exponent = denominator_exponent(r)
    center = r.c
    (x, y) -> log(abs(h(x, y) / (1 + (x - center[1])^2 + (y - center[2])^2)^exponent))
end

function _contour_function(
    r::RoutingFunction;
    h::Union{Function, Nothing} = nothing,
    RR::Union{Function, Nothing} = nothing,
)
    if !isnothing(RR)
        RR
    elseif !isnothing(h)
        _default_contour_function(r, h)
    else
        (x, y) -> real(r([x, y]))
    end
end

function _root_count_at_point(
    pt;
    root_count_fn::Union{Function, Nothing} = nothing,
    root_counting_system::Union{System, Nothing} = nothing,
    root_count_condition::Union{Function, Nothing} = nothing,
)
    if !isnothing(root_count_fn)
        return root_count_fn(pt)
    end

    real_zeros = HC.solve(root_counting_system, target_parameters = pt) |> real_solutions
    filtered_solutions = isnothing(root_count_condition) ? real_zeros : filter(root_count_condition, real_zeros)
    length(filtered_solutions)
end

function root_counts_at_points(
    pts::AbstractVector{<:AbstractVector{<:Real}};
    root_count_fn::Union{Function, Nothing} = nothing,
    root_counting_system::Union{System, Nothing} = nothing,
    root_count_condition::Union{Function, Nothing} = nothing,
)
    if isnothing(root_count_fn) && isnothing(root_counting_system)
        error("A root count source is required. Pass either root_count_fn or root_counting_system.")
    end

    [
        _root_count_at_point(
            pt;
            root_count_fn = root_count_fn,
            root_counting_system = root_counting_system,
            root_count_condition = root_count_condition,
        )
        for pt in pts
    ]
end

function _component_root_counts(
    G::AbstractVector{<:AbstractVector{<:Integer}},
    pts::AbstractVector{<:AbstractVector{<:Real}};
    root_count_fn::Union{Function, Nothing} = nothing,
    root_counting_system::Union{System, Nothing} = nothing,
    root_count_condition::Union{Function, Nothing} = nothing,
)
    [
        root_counts_at_points(
            pts[component];
            root_count_fn = root_count_fn,
            root_counting_system = root_counting_system,
            root_count_condition = root_count_condition,
        )
        for component in G
    ]
end

function _positive_eigenvectors(∇r::RoutingGradient, pt)
    jacobian = real(evaluate_and_jacobian(∇r, pt)[2])
    eigen_data = LinearAlgebra.eigen(jacobian)
    [real(vector) for (value, vector) in zip(eigen_data.values, eachcol(eigen_data.vectors)) if value > 0]
end

function _plot_flow_branch!(pl, flow, flow_breakpoint_ratio, flow_linewidth, arrowstyle)
    if length(flow) < 2
        return pl
    end

    breakpoint = clamp(div(length(flow), flow_breakpoint_ratio), 1, length(flow) - 1)
    plot!(pl, flow[1:breakpoint], linecolor = :steelblue, linewidth = flow_linewidth, label = false, arrow = arrowstyle)
    plot!(pl, flow[breakpoint:end], linecolor = :steelblue, linewidth = flow_linewidth, label = false)
end

function _plot_unstable_flows!(
    pl,
    ∇r::RoutingGradient,
    pts::AbstractVector{<:AbstractVector{<:Real}},
    idx::AbstractVector{<:Integer};
    arrowstyle,
    flow_linewidth,
    flow_breakpoint_ratio,
    flow_seed_scale,
    flow_tspan,
    plot_all_unstable_flows,
)
    g(x, param, t) = real(evaluate(∇r, x))

    for pt in pts[findall(!iszero, idx)]
        unstable_directions = _positive_eigenvectors(∇r, pt)
        if isempty(unstable_directions)
            continue
        end

        directions = plot_all_unstable_flows ? unstable_directions : unstable_directions[1:1]
        for direction in directions
            normalized_direction = direction / norm(direction)
            for sign in (-1.0, 1.0)
                problem = SciMLBase.ODEProblem(
                    g,
                    pt + sign * flow_seed_scale * normalized_direction,
                    flow_tspan,
                )
                solution = DE.solve(problem, reltol = 1e-6, abstol = 1e-6)
                _plot_flow_branch!(pl, Tuple.(solution.u), flow_breakpoint_ratio, flow_linewidth, arrowstyle)
            end
        end
    end

    pl
end

function _annotate_root_counts!(
    pl,
    pts::AbstractVector{<:AbstractVector{<:Real}},
    counts::AbstractVector{<:Integer};
    annotation_formatter::Function,
    annotation_offset,
    annotation_textsize::Integer,
    annotation_color,
)
    dx, dy = annotation_offset
    annotations = [
        text(annotation_formatter(count, pt, index), annotation_textsize, annotation_color)
        for (index, (pt, count)) in enumerate(zip(pts, counts))
    ]
    annotate!(pl, first.(pts) .+ dx, last.(pts) .+ dy, annotations)
end





function generate_plot(
    r::RoutingFunction,
    routing_points::AbstractVector{<:AbstractVector{<:Real}},
    G::AbstractVector{<:AbstractVector{<:Integer}},
    idx::AbstractVector{<:Integer};
    root_counting_system::Union{System, Nothing} = nothing,
    root_count_fn::Union{Function, Nothing} = nothing,
    h::Union{Function, Nothing} = nothing,
    RR::Union{Function, Nothing} = nothing,
    root_count_condition::Union{Function, Nothing} = nothing,
    annotate_root_counts::Union{Bool, Nothing} = nothing,
    annotation_formatter::Function = (count, pt, index) -> string(count),
    annotation_textsize::Integer = 6,
    annotation_offset = (0.0, 0.0),
    annotation_color = :black,
    arrowstyle = :closed,
    markersize = 3,
    legend = false,
    flow_linewidth = 2,
    discriminant_linewidth = 2,
    flow_breakpoint_ratio = 3,
    flow_seed_scale = 0.01,
    flow_tspan = (0.0, 1e4),
    plot_all_unstable_flows = false,
    plot_contour = true,
    contour_stepsize = 0.01,
    contour_levels = 50,
    contour_color = :plasma,
    contour_linewidth = 1,
    xlims = nothing,
    ylims = nothing
)
    ∇r = RoutingGradient(r)

    has_root_count_source = !isnothing(root_count_fn) || !isnothing(root_counting_system)
    should_annotate = isnothing(annotate_root_counts) ? has_root_count_source : annotate_root_counts

    component_counts = nothing
    point_counts = nothing
    if has_root_count_source
        component_counts = _component_root_counts(
            G,
            routing_points;
            root_count_fn = root_count_fn,
            root_counting_system = root_counting_system,
            root_count_condition = root_count_condition,
        )
        point_counts = root_counts_at_points(
            routing_points;
            root_count_fn = root_count_fn,
            root_counting_system = root_counting_system,
            root_count_condition = root_count_condition,
        )

        for (component_index, root_counts) in enumerate(component_counts)
            println("Connected component #$(component_index)")
            if isnothing(root_count_condition)
                println("Real root counts: $(root_counts)\n")
            else
                println("Filtered real root counts: $(root_counts)\n")
            end
        end
    end

    Δx = xlims[2] - xlims[1]
    Δy = ylims[2] - ylims[1]

    contour_function = _contour_function(r; h = h, RR = RR)
    pl = if plot_contour
        contour(
            xlims[1]:contour_stepsize:xlims[2],
            ylims[1]:contour_stepsize:ylims[2],
            contour_function,
            levels = contour_levels,
            color = contour_color,
            clabels = false,
            cbar = false,
            lw = contour_linewidth,
            size = (600, 600*(Δy/Δx)),
        )
    else
        plot()
    end

    if !isnothing(h)
        implicit_plot!(
            pl,
            h;
            xlims = xlims,
            ylims = ylims,
            linecolor = :black,
            linewidth = discriminant_linewidth,
            label = "Discriminant",
            legend = false,
            resolution = 3000,
        )
    end

    _plot_unstable_flows!(
        pl,
        ∇r,
        routing_points,
        idx;
        arrowstyle = arrowstyle,
        flow_linewidth = flow_linewidth,
        flow_breakpoint_ratio = flow_breakpoint_ratio,
        flow_seed_scale = flow_seed_scale,
        flow_tspan = flow_tspan,
        plot_all_unstable_flows = plot_all_unstable_flows,
    )

    idx0 = findall(iszero, idx)
    idx1 = findall(!iszero, idx)
    !isempty(idx0) && scatter!(pl, Tuple.(routing_points[idx0]), markercolor = "#66C34F", markersize = markersize, label = "Routing point (index 0)")
    !isempty(idx1) && scatter!(pl, Tuple.(routing_points[idx1]), markercolor = "#66C34F", markersize = markersize, marker = :diamond, label = "Routing point (index > 0)")

    if should_annotate
        _annotate_root_counts!(
            pl,
            routing_points,
            point_counts;
            annotation_formatter = annotation_formatter,
            annotation_offset = annotation_offset,
            annotation_textsize = annotation_textsize,
            annotation_color = annotation_color,
        )
    end

    plot!(
        pl;
        xlims = xlims,
        ylims = ylims,
        legend = legend,
        dpi = 400,
        legendfontsize = 6,
    )
end
