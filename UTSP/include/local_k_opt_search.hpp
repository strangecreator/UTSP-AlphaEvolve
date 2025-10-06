#pragma once

// standart imports
#include <cmath>
#include <iostream>


void calc_potential_inplace(const Config& config, Context& context, int i, int j) {
    if (i == j) {
        context.potential[i * config.cities_number + j] = 0.0;
        return;
    }

    double weight = context.weight[i * config.cities_number + j];

    // this forms a smooth ReLU function
    if (weight < 0) { weight = pow(e, weight); }
    else { weight += 1.0; }

    context.potential[i * config.cities_number + j] = weight + config.exploration_coefficient * sqrt(log(context.total_simulations + 1) / (context.chosen_times[i * config.cities_number + j] + 1));  // always a positive value
}

int get_candidate_proportionally_by_potential(const Config& config, Context& context, int current_city, int start_city) {
    int next_city = context.path[current_city].next;

	double total_potential = 0.0;
    int candidates_available = 0;

	for (int i = 0; i < config.candidates_number; ++i) {
        int candidate = context.candidates[current_city * config.candidates_number + i];
        double potential = context.potential[current_city * config.cities_number + candidate];

        if (candidate == next_city || candidate == start_city || (potential < config.min_potential_to_consider)) { continue; }  // not available

		total_potential += potential;
        ++candidates_available;
	}

    if (candidates_available == 0) { return null; }

    // choosing the random available candidate proportionally
    double random_potential = (static_cast<double>(rand()) / RAND_MAX) * total_potential;

	for (int i = 0; i < config.candidates_number; ++i) {
        int candidate = context.candidates[current_city * config.candidates_number + i];
        double potential = context.potential[current_city * config.cities_number + candidate];

        if (candidate == next_city || candidate == start_city || (potential < config.min_potential_to_consider)) { continue; }  // not available

        random_potential -= potential;
        if (random_potential <= 0) { return candidate; }
    }

	return null;
}


bool apply_k_opt_move(const Config& config, Context& context, int start_city, int max_k_opt_depth) {
    ++context.total_simulations;

    // first pair
    int next_to_start_city = context.path[start_city].next;

    context.pairs[0] = start_city;
    context.pairs[1] = next_to_start_city;
    int depth = 1;

    // breaking an edge of the first pair
    context.path[start_city].next = null;
    context.path[next_to_start_city].prev = null;

    // initializing gains
    double gain_double; double gain_double_with_closure = 0.0;
    int gain_int32; int gain_int32_with_closure = 0;
    long long gain_int64; long long gain_int64_with_closure = 0;

    if (config.distance_type == DistanceType::Double) {
        gain_double = get_distance_double(config, context, start_city, next_to_start_city);
    }
    if (config.distance_type == DistanceType::Int32) {
        gain_int32 = get_distance_int32(config, context, start_city, next_to_start_city);
    }
    if (config.distance_type == DistanceType::Int64) {
        gain_int64 = get_distance_int64(config, context, start_city, next_to_start_city);
    }

    bool apply_move = false;
    double weight_increase = 0.0;

    int current_city = next_to_start_city;

    for (int i = 1; i < max_k_opt_depth; ++i) {
        int proposed_city = get_candidate_proportionally_by_potential(config, context, current_city, start_city);

        if (proposed_city == null) { return false; }  // no candidates, could not improve

        ++context.chosen_times[current_city * config.cities_number + proposed_city];
		++context.chosen_times[proposed_city * config.cities_number + current_city];

        int proposed_city_link = context.path[proposed_city].prev;  // city to disconnect from the proposed city (and maybe to connect to the start city)

        context.pairs[2 * i] = proposed_city;
        context.pairs[2 * i + 1] = proposed_city_link;
        ++depth;

        // applying 2 opt move
        reverse_sub_path(context, current_city, proposed_city_link);

        context.path[current_city].next = proposed_city;
        context.path[proposed_city].prev = current_city;
        context.path[proposed_city_link].prev = null;

        // recalculating gains
        if (config.distance_type == DistanceType::Double) {
            gain_double += get_distance_double(config, context, proposed_city_link, proposed_city) - get_distance_double(config, context, current_city, proposed_city);
            gain_double_with_closure = gain_double - get_distance_double(config, context, start_city, proposed_city_link);

            weight_increase = config.weight_delta_coefficient * (pow(e, gain_double_with_closure / context.path_distance_double) - 1);

            if (gain_double_with_closure > 0.0) {
                apply_move = true;
                context.path_distance_double -= gain_double_with_closure;
            }
        }
        if (config.distance_type == DistanceType::Int32) {
            gain_int32 += get_distance_int32(config, context, proposed_city_link, proposed_city) - get_distance_int32(config, context, current_city, proposed_city);
            gain_int32_with_closure = gain_int32 - get_distance_int32(config, context, start_city, proposed_city_link);

            weight_increase = config.weight_delta_coefficient * (pow(e, static_cast<double>(gain_int32_with_closure) / context.path_distance_int32) - 1);

            if (gain_int32_with_closure > 0) {
                apply_move = true;
                context.path_distance_int32 -= gain_int32_with_closure;
            }
        }
        if (config.distance_type == DistanceType::Int64) {
            gain_int64 += get_distance_int64(config, context, proposed_city_link, proposed_city) - get_distance_int64(config, context, current_city, proposed_city);
            gain_int64_with_closure = gain_int64 - get_distance_int64(config, context, start_city, proposed_city_link);

            weight_increase = config.weight_delta_coefficient * (pow(e, static_cast<double>(gain_int64_with_closure) / context.path_distance_int64) - 1);

            if (gain_int64_with_closure > 0) {
                apply_move = true;
                context.path_distance_int64 -= gain_int64_with_closure;
            }
        }

        if (apply_move) { break; }

        current_city = proposed_city_link;
    }

    // updating weights
    for (int i = 0; i < depth; ++i) {
        int current_city = context.pairs[2 * i];
        int proposed_city = (i < depth - 1) ? context.pairs[2 * i + 2] : start_city;

        double factor = 1.0;
        if (!apply_move) {
            // exponential decrease in sensitivity
            factor = pow(e, -i / config.sensitivity_temperature);
        }

        context.weight[current_city * config.cities_number + proposed_city] += weight_increase * factor;
        context.weight[proposed_city * config.cities_number + current_city] += weight_increase * factor;
    }

    if (apply_move) {
        int end_city = context.pairs[2 * depth - 1];

        context.path[start_city].next = end_city;
        context.path[end_city].prev = start_city;

        return true;
    }
    return false;
}


bool improve_by_k_opt_move(const Config& config, Context& context, int max_k_opt_depth) {
    for (int i = 0; i < config.max_k_opt_simulations_without_improve_to_stop; ++i) {
        // saving current path
        convert_path_to_solution(config, context);

        int start_city = get_random_int_by_module(config.cities_number);
        if (apply_k_opt_move(config, context, start_city, max_k_opt_depth)) { return true; }

        // restoring the path that was before the move
        convert_solution_to_path(config, context);
    }

    return false;
}

int local_k_opt_search(const Config& config, Context& context, int max_k_opt_depth) {
    // calculating current potentials for each edge
    for (int i = 0; i < config.cities_number; ++i) {
        for (int j = 0; j < config.cities_number; ++j) {
            calc_potential_inplace(config, context, i, j);
        }
    }

    // running simulations and trying to improve
    int improved_times = 0;

    while (improve_by_k_opt_move(config, context, max_k_opt_depth)) { ++improved_times; };

    return improved_times;
}