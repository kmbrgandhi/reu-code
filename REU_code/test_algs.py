
import time
from evolut_full import evolut_algo, evolut_algo_naive
from conv_utils import rand_select_algorithm_doppler, gen_rand_seq, gen_binary_codes, gen_polyphase_codes
from genetic_full import run_evolution, select_parents_part_random, select_parents_random, tournament_selection
from crossover_algs import crossover_halfhalf, crossover_random, crossover_randpoint
from hill_climbing import simple_local_search, stochastic_local_search, hill_climbing_iterative, sdls_local_search
import pandas as pd
from anneal_full import anneal_mult, anneal, anneal_with_tabu, anneal_with_local_search, semi_exponential, lin_add, lin_mult, mult_exponential, log_mult, quad_mult, quad_add, expon_add, trigon_add, quenching
from memetic_tabu_search import run_evolution_memet, ts_local_search
from threshold_accept import threshold_handler
from polyphase import great_deluge_handler, anneal_phases_handler, random_arbitrary_polyphase
from particle_swarm import particle_swarm_polyphase
from alternating_codes import hill_climbing_iterative_alternating, great_deluge_alternating, random_alternating_binary, random_alternating_polyphase
from exhaustive import peak_sidelobes_general, peak_sidelobes_efficient, s_doppler


# BINARY TESTS:
# These tests aim to minimize peak sidelobe values for binary sidelobes for codes of length 2-55.
# This checks how well my algorithms do relative to the optimal possible, as well as compared to each other.
# Note that we also have a randomized test at the end, which aims to check how well we do relative to a random algorithm
# with the same time constraints.

def hill_climbing_tests():
    print('hills')
    f = open('best_known_sidelobes.txt')
    lines = f.readlines()
    steepest_descent = ['NA', 'NA']
    simple_climb = ['NA', 'NA']
    stochastic_climb = ['NA', 'NA']
    for i in range(2, 55):
        print(i)
        x = hill_climbing_iterative(i, sdls_local_search)
        steepest_descent.append(x[1].real - int(lines[i-1]))
        y = hill_climbing_iterative(i, simple_local_search)
        simple_climb.append(y[1].real - int(lines[i-1]))
        z = hill_climbing_iterative(i, stochastic_local_search)
        stochastic_climb.append(z[1].real - int(lines[i-1]))
    df = pd.DataFrame({'Steepest Descent': steepest_descent, 'Simple Climb': simple_climb, 'Stochastic Climb': stochastic_climb})
    writer = pd.ExcelWriter('hill_climb.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def anneal_tests():
    print('anneal')
    f = open('best_known_sidelobes.txt')
    lines = f.readlines()
    anneal_reg = ['NA', 'NA']
    anneal_local = ['NA', 'NA']
    anneal_tabu = ['NA', 'NA']
    for i in range(2, 55):
        print(i)
        x = anneal_mult(i, 1, semi_exponential, anneal)
        anneal_reg.append(x.real - int(lines[i-1]))
        y = anneal_mult(i, 1, semi_exponential, anneal_with_tabu)
        anneal_tabu.append(y.real - int(lines[i-1]))

    print(anneal_reg)
    print(anneal_tabu)
    df = pd.DataFrame({'Regular Anneal': anneal_reg, 'Anneal with tabu': anneal_tabu})
    writer = pd.ExcelWriter('anneal_test_binary.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def evolutionary_tests():
    print('evolution')
    f = open('best_known_sidelobes.txt')
    lines = f.readlines()
    evolut_smart = ['NA', 'NA']
    evolut_naive = ['NA', 'NA']
    for i in range(2, 55):
        print(i)
        x = evolut_algo_naive(i)
        evolut_smart.append(x[1].real - int(lines[i - 1]))
        y = evolut_algo_naive(i)
        evolut_naive.append(y[1].real - int(lines[i - 1]))


    df = pd.DataFrame(
        {'Smart evolutionary': evolut_smart, 'Naive evolutionary': evolut_naive})
    writer = pd.ExcelWriter('evolutionary_test_binary.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def genetic_rand_crossover_tests():
    print('genetic_rand')
    f = open('best_known_sidelobes.txt')
    lines = f.readlines()
    genetic_rand_rand = ['NA', 'NA']
    genetic_rand_partrand = ['NA', 'NA']
    genetic_rand_tourney = ['NA', 'NA']
    for i in range(2, 55):
        print(i)
        x = run_evolution(i, parent_func = select_parents_random, crossover_func = crossover_random)
        genetic_rand_rand.append(x[1].real - int(lines[i - 1]))
        y = run_evolution(i, parent_func = select_parents_part_random, crossover_func = crossover_random)
        genetic_rand_partrand.append(y[1].real - int(lines[i - 1]))
        z = run_evolution(i, parent_func = tournament_selection, crossover_func = crossover_random)
        genetic_rand_tourney.append(z[1].real - int(lines[i - 1]))

    df = pd.DataFrame(
        {'Random crossover with random parent selection': genetic_rand_rand, 'Random crossover with partially random parent selection': genetic_rand_partrand,
         'Random crossover with tournament parent selection': genetic_rand_tourney})
    writer = pd.ExcelWriter('genetic_random_crossover_test_binary.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def genetic_randpoint_crossover_tests():
    print('genetic_randpoint')
    f = open('best_known_sidelobes.txt')
    lines = f.readlines()
    genetic_randpt_rand = ['NA', 'NA']
    genetic_randpt_partrand = ['NA', 'NA']
    genetic_randpt_tourney = ['NA', 'NA']
    for i in range(2, 55):
        print(i)
        x = run_evolution(i, parent_func=select_parents_random, crossover_func=crossover_randpoint)
        genetic_randpt_rand.append(x[1].real - int(lines[i - 1]))
        y = run_evolution(i, parent_func=select_parents_part_random, crossover_func=crossover_randpoint)
        genetic_randpt_partrand.append(y[1].real - int(lines[i - 1]))
        z = run_evolution(i, parent_func=tournament_selection, crossover_func=crossover_randpoint)
        genetic_randpt_tourney.append(z[1].real - int(lines[i - 1]))

    df = pd.DataFrame(
        {'Random point crossover with random parent selection': genetic_randpt_rand, 'Random point crossover with partially random parent selection': genetic_randpt_partrand,
         'Random point crossover with tournament parent selection': genetic_randpt_tourney})
    writer = pd.ExcelWriter('genetic_randpoint_crossover_test_binary.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def genetic_half_crossover_tests():
    print('genetic_half')
    f = open('best_known_sidelobes.txt')
    lines = f.readlines()
    genetic_half_rand = ['NA', 'NA']
    genetic_half_partrand = ['NA', 'NA']
    genetic_half_tourney = ['NA', 'NA']
    for i in range(2, 55):
        print(i)
        x = run_evolution(i, parent_func=select_parents_random, crossover_func=crossover_randpoint)
        genetic_half_rand.append(x[1].real - int(lines[i - 1]))
        y = run_evolution(i, parent_func=select_parents_part_random, crossover_func=crossover_randpoint)
        genetic_half_partrand.append(y[1].real - int(lines[i - 1]))
        z = run_evolution(i, parent_func=tournament_selection, crossover_func=crossover_randpoint)
        genetic_half_tourney.append(z[1].real - int(lines[i - 1]))

    df = pd.DataFrame(
        {'Half crossover with random parent selection': genetic_half_rand, 'Half crossover with partially random parent selection': genetic_half_partrand,
         'Half crossover with tournament parent selection': genetic_half_tourney})
    writer = pd.ExcelWriter('genetic_half_crossover_test_binary.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def threshold_accept_tests():
    print('threshold')
    f = open('best_known_sidelobes.txt')
    lines = f.readlines()
    threshold = ['NA', 'NA']
    for i in range(2, 55):
        print(i)
        x = threshold_handler(i)
        threshold.append(x.real - int(lines[i - 1]))
    df = pd.DataFrame({'Threshold accept': threshold})
    writer = pd.ExcelWriter('threshold_accept_test_binary.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def memetic_tests():
    print('memetic')
    f = open('best_known_sidelobes.txt')
    lines = f.readlines()
    memetic_ts = ['NA', 'NA']
    memetic_sdls= ['NA', 'NA']
    memetic_simple = ['NA', 'NA']
    for i in range(2, 55):
        print(i)
        x = run_evolution_memet(i, ts_local_search)
        memetic_ts.append(x[1].real - int(lines[i - 1]))
        y = run_evolution_memet(i, sdls_local_search)
        memetic_sdls.append(y[1].real - int(lines[i - 1]))
        z = run_evolution_memet(i, simple_local_search)
        memetic_simple.append(z[1].real - int(lines[i - 1]))
    df = pd.DataFrame(
        {'Memetic w/ tabu search': memetic_ts, 'Memetic with sdls': memetic_sdls, 'Memetic with simple': memetic_simple})
    writer = pd.ExcelWriter('memetic_test_binary.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def random_tests():
    print('random')
    f = open('best_known_sidelobes.txt')
    lines = f.readlines()
    random_vals = ['NA', 'NA']
    for i in range(2, 55):
        print(i)
        x = rand_select_algorithm_doppler(i, 2, 1)
        random_vals.append(x[1].real - int(lines[i - 1]))
    df = pd.DataFrame({'Random': random_vals})
    writer = pd.ExcelWriter('random_test_binary.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

# RANDOM POLYPHASE TESTS:
# These are tests, with codes chosen randomly at each step, which aim to minimize the psl values for polyphase codes
# with variable doppler widths.  These serve as a control against other tests we will run with the memetic and
# hill climbing algorithms.

lst = [2, 3, 4, 6, 16]
freqs = [1, 2, 3, 4, 6]

def random_polyphase_N1_tests():
    print('random_N1')
    lengths = []
    phase_unity = []
    random_codes = []
    random_vals = []
    for i in range(5, 41, 5):
        print(i)
        for j in lst:
            print(j)
            lengths.append(i)
            phase_unity.append(j)
            x = rand_select_algorithm_doppler(i, j, 1)
            random_codes.append(x[0])
            random_vals.append(x[1])
            print(x[1])
    df = pd.DataFrame({'Length': lengths, 'Phase Unity': phase_unity, 'Code': random_codes, 'PSL Value': random_vals})
    writer = pd.ExcelWriter('random_test_polyphase_N1.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def random_polyphase_N2_tests():
    print('random_N2')
    lengths = []
    phase_unity = []
    random_codes = []
    random_vals = []
    for i in range(5, 41, 5):
        print(i)
        for j in lst:
            print(j)
            lengths.append(i)
            phase_unity.append(j)
            x = rand_select_algorithm_doppler(i, j, 2)
            random_codes.append(x[0])
            random_vals.append(x[1])
    df = pd.DataFrame({'Length': lengths, 'Phase Unity': phase_unity, 'Code': random_codes, 'PSL Value': random_vals})
    writer = pd.ExcelWriter('random_test_polyphase_N2.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def random_polyphase_N3_tests():
    print('random_N3')
    lengths = []
    phase_unity = []
    random_codes = []
    random_vals = []
    for i in range(5, 41, 5):
        print(i)
        for j in lst:
            print(j)
            lengths.append(i)
            phase_unity.append(j)
            x = rand_select_algorithm_doppler(i, j, 3)
            random_codes.append(x[0])
            random_vals.append(x[1])
    df = pd.DataFrame({'Length': lengths, 'Phase Unity': phase_unity, 'Code': random_codes, 'PSL Value': random_vals})
    writer = pd.ExcelWriter('random_test_polyphase_N3.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def random_polyphase_N4_tests():
    print('random_N4')
    lengths = []
    phase_unity = []
    random_codes = []
    random_vals = []
    for i in range(5, 41, 5):
        print(i)
        for j in lst:
            print(j)
            lengths.append(i)
            phase_unity.append(j)
            x = rand_select_algorithm_doppler(i, j, 4)
            random_codes.append(x[0])
            random_vals.append(x[1])
    df = pd.DataFrame({'Length': lengths, 'Phase Unity': phase_unity, 'Code': random_codes, 'PSL Value': random_vals})
    writer = pd.ExcelWriter('random_test_polyphase_N4.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def random_polyphase_N6_tests():
    print('random_N6')
    lengths = []
    phase_unity = []
    random_codes = []
    random_vals = []
    for i in range(5, 41, 5):
        print(i)
        for j in lst:
            print(j)
            lengths.append(i)
            phase_unity.append(j)
            x = rand_select_algorithm_doppler(i, j, 6)
            random_codes.append(x[0])
            random_vals.append(x[1])
    df = pd.DataFrame({'Length': lengths, 'Phase Unity': phase_unity, 'Code': random_codes, 'PSL Value': random_vals})
    writer = pd.ExcelWriter('random_test_polyphase_N6.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

# TEMPERATURE TESTS:
# This one, long test compares the various temperature schedules, timing them and checking for various levels of optimization.
# If rewritten, this test will need to be a little more stringent about timing constraints to make sure that none of these
# take excessively long inside a loop, thus escaping constraints.

lst = [8, 13, 19, 25, 29, 35]
def temperature_tests():
    print('temperature')
    f = open('best_known_sidelobes.txt')
    lines = f.readlines()
    lengths = []
    anneal_semiexponential = []
    anneal_semiexponential_times = []
    anneal_mult_exponential = []
    anneal_mult_exponential_times = []
    anneal_log_mult = []
    anneal_log_mult_times = []
    anneal_lin_mult = []
    anneal_lin_mult_times = []
    anneal_quad_mult = []
    anneal_quad_mult_times = []
    anneal_lin_add = []
    anneal_lin_add_times = []
    anneal_quad_add = []
    anneal_quad_add_times = []
    anneal_expon_add = []
    anneal_expon_add_times = []
    anneal_trigon_add = []
    anneal_trigon_add_times = []
    anneal_quenching = []
    anneal_quenching_times = []
    for i in lst:
        print(i)
        lengths.append(i)
        start_time = time.time()
        x = anneal_mult(i, 1, semi_exponential, anneal)
        anneal_semiexponential_times.append(time.time()-start_time)
        anneal_semiexponential.append(x.real)
        if i == 35:
            print('checkpoint 1')
        start_time = time.time()
        x = anneal_mult(i, 1, mult_exponential, anneal)
        anneal_mult_exponential_times.append(time.time() - start_time)
        anneal_mult_exponential.append(x.real)
        if i == 35:
            print('checkpoint 2')
        start_time = time.time()
        x = anneal_mult(i, 1, log_mult, anneal)
        anneal_log_mult_times.append(time.time() - start_time)
        anneal_log_mult.append(x.real)
        if i == 35:
            print('checkpoint 3')
        start_time = time.time()
        x = anneal_mult(i, 1, lin_mult, anneal)
        anneal_lin_mult_times.append(time.time() - start_time)
        anneal_lin_mult.append(x.real)
        if i == 35:
            print('checkpoint 4')
        start_time = time.time()
        x = anneal_mult(i, 1, quad_mult, anneal)
        anneal_quad_mult_times.append(time.time() - start_time)
        anneal_quad_mult.append(x.real)
        if i == 35:
            print('checkpoint 5')
        start_time = time.time()
        x = anneal_mult(i, 1, lin_add, anneal)
        print(x)
        anneal_lin_add_times.append(time.time() - start_time)
        anneal_lin_add.append(x.real)
        if i == 35:
            print('checkpoint 6')
        start_time = time.time()
        x = anneal_mult(i, 1, quad_add, anneal)
        anneal_quad_add_times.append(time.time() - start_time)
        anneal_quad_add.append(x.real)
        if i == 35:
            print('checkpoint 7')
        start_time = time.time()
        x = anneal_mult(i, 1, expon_add, anneal)
        anneal_expon_add_times.append(time.time() - start_time)
        anneal_expon_add.append(x.real)
        if i == 35:
            print('checkpoint 8')
        start_time = time.time()
        x = anneal_mult(i, 1, trigon_add, anneal)
        anneal_trigon_add_times.append(time.time() - start_time)
        anneal_trigon_add.append(x.real)
        if i == 35:
            print('checkpoint 9')
        start_time = time.time()
        x = anneal_mult(i, 1, quenching, anneal)
        anneal_quenching_times.append(time.time() - start_time)
        anneal_quenching.append(x.real)

    df = pd.DataFrame(
        {'Length': lengths, 'Semiexponential': anneal_semiexponential, 'Semiexponential times': anneal_semiexponential_times,
         'Mult_exponential': anneal_mult_exponential,'Mult_exponential times': anneal_mult_exponential_times,
         'Log_mult': anneal_log_mult, 'Log_mult times': anneal_log_mult_times,
         'Lin_mult': anneal_lin_mult, 'Lin_mult times': anneal_lin_mult_times,  'Quad_mult': anneal_quad_mult,
         'Quad_mult times': anneal_quad_mult_times,
         'Lin_add': anneal_lin_add, 'Lin_add times': anneal_lin_add_times,  'Quad_add': anneal_quad_add,
         'Quad_add times': anneal_quad_add_times, 'Expon_add': anneal_expon_add,
         'Expon_add times': anneal_expon_add_times, 'Trigon_add': anneal_trigon_add,
         'Trigon_add times': anneal_trigon_add_times, 'Quenching': anneal_quenching,
         'Quenching times': anneal_quenching_times})
    writer = pd.ExcelWriter('temperature_tests.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

# DOPPLER, POLYPHASE TESTS:
# This set of tests runs our optimization algorithms to find minimal psl codes with N>1 doppler width, j>2 phase_unity.
# First, we run a hill_climbing test in which our timing is very short, but we get near-optimal solutions for all
# lengths between 2 and 55, and all doppler widths between 1-5, phase_unity between 2 and 5.
# Then, we run our hill_climbing and memetic algorithms on the exact same tests as those for the random polyphase tests above,
# thus serving as a nice point of comparison.

def hill_climbing_doppler_polyphase_tests():
    print('hill_climbing_doppler_polyphase')
    lengths = []
    freq_bands = []
    phase_unity = []
    hill_climbing_codes = []
    hill_climbing_values = []
    times = []
    for j in range(2, 6):
        for N in range(1, 6):
            if j !=2 or N!=1:
                for length in range(2, 55):
                    print(length)
                    print(j)
                    print(N)
                    start_time = time.time()
                    lengths.append(length)
                    freq_bands.append(N)
                    phase_unity.append(j)
                    x = hill_climbing_iterative(length, sdls_local_search, N, j)
                    hill_climbing_codes.append(x[0])
                    hill_climbing_values.append(x[1])
                    times.append(time.time() - start_time)
                    print(x[1])
    df = pd.DataFrame({'Length': lengths, 'Doppler Width': freq_bands,
                       'Phase Unity': phase_unity, 'Code': hill_climbing_codes, 'PSL Value': hill_climbing_values, 'Time': times})
    writer = pd.ExcelWriter('hill_climbing_doppler_polyphase_tests.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def hill_climbing_doppler_binary_select_tests():
    print('hill_climbing_doppler_polyphase')
    lengths = []
    freq_bands = []
    hill_climbing_codes = []
    hill_climbing_values = []
    times = []
    for N in range(2, 6):
        for length in range(5, 41, 5):
            print(length)
            print(N)
            start_time = time.time()
            lengths.append(length)
            freq_bands.append(N)
            x = hill_climbing_iterative(length, sdls_local_search, N, 2)
            hill_climbing_codes.append(x[0])
            hill_climbing_values.append(x[1])
            times.append(time.time() - start_time)
            print(x[1])
    df = pd.DataFrame({'Length': lengths, 'Doppler Width': freq_bands,
                       'Code': hill_climbing_codes, 'PSL Value': hill_climbing_values, 'Time': times})
    writer = pd.ExcelWriter('hill_climbing_doppler_binary_select_tests.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def memetic_doppler_binary_select_tests():
    print('memetic_doppler_binary_select')
    lengths = []
    freq_bands = []
    memetic_codes = []
    memetic_values = []
    times = []
    for N in range(2, 6):
        for length in range(5, 41, 5):
            print(length)
            print(N)
            start_time = time.time()
            lengths.append(length)
            freq_bands.append(N)
            x = run_evolution_memet(length, ts_local_search, 0.1, N)
            memetic_codes.append(x[0])
            memetic_values.append(x[1])
            times.append(time.time() - start_time)
            print(x[1])
    df = pd.DataFrame({'Length': lengths, 'Doppler Width': freq_bands,
                       'Code': memetic_codes, 'PSL Value': memetic_values, 'Time': times})
    writer = pd.ExcelWriter('memetic_doppler_binary_select_tests.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def hill_climbing_doppler_polyphase_select_tests():
    print('hill_climbing_doppler_polyphase_select')
    lengths = []
    freq_bands = []
    hill_climbing_codes = []
    hill_climbing_values = []
    phase_unity = []
    times = []
    for j in range(3, 7):
        for N in range(1, 3):
            for length in range(5, 41, 5):
                print(length)
                print(N)
                print(j)
                start_time = time.time()
                lengths.append(length)
                freq_bands.append(N)
                phase_unity.append(j)
                x = hill_climbing_iterative(length, sdls_local_search, N, j)
                hill_climbing_codes.append(x[0])
                hill_climbing_values.append(x[1])
                times.append(time.time() - start_time)
                print(x[1])
    df = pd.DataFrame({'Length': lengths, 'Doppler Width': freq_bands, 'Phase Unity': phase_unity,
                       'Code': hill_climbing_codes, 'PSL Value': hill_climbing_values, 'Time': times})
    writer = pd.ExcelWriter('hill_climbing_doppler_polyphase_select_tests.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def memetic_doppler_polyphase_select_tests():
    print('memetic_doppler_polyphase_select')
    lengths = []
    freq_bands = []
    memetic_codes = []
    memetic_values = []
    phase_unity = []
    times = []
    for j in range(3, 7):
        for N in range(1, 3):
            for length in range(5, 41, 5):
                print(length)
                print(N)
                print(j)
                start_time = time.time()
                lengths.append(length)
                freq_bands.append(N)
                phase_unity.append(j)
                x = run_evolution_memet(length, ts_local_search, 0.1, N, j)
                memetic_codes.append(x[0])
                memetic_values.append(x[1])
                times.append(time.time() - start_time)
                print(x[1])
    df = pd.DataFrame({'Length': lengths, 'Doppler Width': freq_bands, 'Phase Unity': phase_unity,
                       'Code': memetic_codes, 'PSL Value': memetic_values, 'Time': times})
    writer = pd.ExcelWriter('memetic_doppler_polyphase_select_tests.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

# ARBITRARY POLYPHASE TESTS
# These tests, for lengths 2-39, aim to find optimal psl arbitrary polyphase codes with N=1 doppler width.
# They also compare the relative success of random, great deluge, simulated annealing, and particle swarm algorithms.
# After this comparison, we then run doppler tests with the best polyphase algorithm found, generating some new
# optimal codes in doppler space.
def random_arbitrary_polyphase_test():
    print('random arbitrary polyphase')
    length = []
    random_codes = []
    random_values = []
    for i in range(2, 39):
        print(i)
        length.append(i)
        x = random_arbitrary_polyphase(i)
        random_codes.append(x[0])
        random_values.append(x[1])
        print(x[1])

    df = pd.DataFrame({'Length': length,
                       'Code': random_codes, 'PSL Value': random_values})
    writer = pd.ExcelWriter('random_arbitrary_polyphase.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def great_deluge_tests():
    print('great deluge polyphase')
    great_deluge = ['NA', 'NA']
    for i in range(2, 39):
        print(i)
        x = great_deluge_handler(i)
        great_deluge.append(max(0, x[1].real - 1))
    df = pd.DataFrame({'Great Deluge': great_deluge})
    writer = pd.ExcelWriter('great_deluge_test_arb_polyphase.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


def anneal_phase_tests():
    print('anneal polyphase')
    anneal = ['NA', 'NA']
    for i in range(2, 39):
        print(i)
        x = anneal_phases_handler(i, semi_exponential)
        anneal.append(max(0, x[1].real - 1))
    df = pd.DataFrame({'Anneal': anneal})
    writer = pd.ExcelWriter('anneal_test_arb_polyphase.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def particle_swarm_tests():
    print('particle polyphase')
    particle_swarm = ['NA', 'NA']
    for i in range(2, 39):
        print(i)
        x = particle_swarm_polyphase(i)
        particle_swarm.append(max(0, x[1].real - 1))
    df = pd.DataFrame({'Particle Swarm': particle_swarm})
    writer = pd.ExcelWriter('particle_swarm_test_arb_polyphase.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def best_polyphase_gen_codes_N1_tests():
    print('best polyphase')
    lengths = []
    great_deluge_codes = []
    great_deluge_values = []
    for i in range(5, 46, 5):
        print(i)
        lengths.append(i)
        x = great_deluge_handler(i)
        great_deluge_codes.append(x[0])
        print(x[1])
        great_deluge_values.append(x[1])
    df = pd.DataFrame({'Length': lengths, 'Code': great_deluge_codes, 'Value': great_deluge_values})
    writer = pd.ExcelWriter('best_polyphase_N1.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def best_polyphase_gen_codes_N2_tests():
    print('best polyphase 2')
    lengths = []
    great_deluge_codes = []
    great_deluge_values = []
    for i in range(5, 46, 5):
        print(i)
        lengths.append(i)
        x = great_deluge_handler(i, N=2)
        great_deluge_codes.append(x[0])
        great_deluge_values.append(x[1])
    df = pd.DataFrame({'Length': lengths, 'Code': great_deluge_codes, 'Value': great_deluge_values})
    writer = pd.ExcelWriter('best_polyphase_N2.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def best_polyphase_gen_codes_N3_tests():
    print('best polyphase 3')
    lengths = []
    great_deluge_codes = []
    great_deluge_values = []
    for i in range(5, 46, 5):
        print(i)
        lengths.append(i)
        x = great_deluge_handler(i, N=3)
        great_deluge_codes.append(x[0])
        great_deluge_values.append(x[1])
    df = pd.DataFrame({'Length': lengths, 'Code': great_deluge_codes, 'Value': great_deluge_values})
    writer = pd.ExcelWriter('best_polyphase_N3.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def best_polyphase_gen_codes_N4_tests():
    print('best polyphase 4')
    lengths = []
    great_deluge_codes = []
    great_deluge_values = []
    for i in range(5, 46, 5):
        print(i)
        lengths.append(i)
        x = great_deluge_handler(i, N=4)
        great_deluge_codes.append(x[0])
        great_deluge_values.append(x[1])
    df = pd.DataFrame({'Length': lengths, 'Code': great_deluge_codes, 'Value': great_deluge_values})
    writer = pd.ExcelWriter('best_polyphase_N4.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def best_polyphase_gen_codes_N5_tests():
    print('best polyphase 5')
    lengths = []
    great_deluge_codes = []
    great_deluge_values = []
    for i in range(5, 46, 5):
        print(i)
        lengths.append(i)
        x = great_deluge_handler(i, N=5)
        great_deluge_codes.append(x[0])
        great_deluge_values.append(x[1])
    df = pd.DataFrame({'Length': lengths, 'Code': great_deluge_codes, 'Value': great_deluge_values})
    writer = pd.ExcelWriter('best_polyphase_N5.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def best_polyphase_gen_codes_N6_tests():
    print('best polyphase 6')
    lengths = []
    great_deluge_codes = []
    great_deluge_values = []
    for i in range(5, 46, 5):
        print(i)
        lengths.append(i)
        x = great_deluge_handler(i, N=6)
        great_deluge_codes.append(x[0])
        great_deluge_values.append(x[1])
    df = pd.DataFrame({'Length': lengths, 'Code': great_deluge_codes, 'Value': great_deluge_values})
    writer = pd.ExcelWriter('best_polyphase_N6.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

# EXHAUSTIVE TESTS:
# These tests aim to test our exhaustive algorithms, and in particular demonstrate the efficiency of our efficient exhaustive
# method as compared to naive methods.  This also serves as a (start to) a point of comparison for optimal values for
# polyphase codes with doppler width.
def naive_exhaustive_tests():
    print('naive')
    lengths = []
    freq_bands = []
    phase_unity = []
    naive_codes = []
    naive_values = []
    times = []
    for j in range(2, 6):
        for N in range(1, 6):
            count = 2
            start_time = time.time()
            while time.time() - start_time < 600:
                print(count)
                print(j)
                print(N)
                start_time = time.time()
                lengths.append(count)
                freq_bands.append(N)
                phase_unity.append(j)
                if j == 2:
                    x = peak_sidelobes_general(count, None, N, gen_binary_codes)
                else:
                    x = peak_sidelobes_general(count, None, N, gen_polyphase_codes, optional_limit = j)

                naive_codes.append(x[0])
                naive_values.append(x[1])
                times.append(time.time() - start_time)
                print(x[1])
                count+=1
    df = pd.DataFrame({'Length': lengths, 'Doppler Width': freq_bands,
                       'Phase Unity': phase_unity, 'Code': naive_codes, 'PSL Value': naive_values, 'Time': times})
    writer = pd.ExcelWriter('naive_exhaustive.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def efficient_nondoppler_tests():
    print('efficient')
    lengths = []
    freq_bands = []
    efficient_codes = []
    efficient_values = []
    times = []
    for N in range(1, 6):
        count = 2
        start_time = time.time()
        while time.time() - start_time < 450:
            print(count)
            print(N)
            start_time = time.time()
            lengths.append(count)
            freq_bands.append(N)
            if N==1:
                x = peak_sidelobes_efficient(count)
            else:
                x = peak_sidelobes_efficient(count, s_doppler, 0, float('inf'), N)
            efficient_codes.append(x[0])
            efficient_values.append(x[1])
            times.append(time.time() - start_time)
            print(x[1])
            count += 1
    df = pd.DataFrame({'Length': lengths, 'Doppler Width': freq_bands,
                     'Code': efficient_codes, 'PSL Value': efficient_values, 'Time': times})
    writer = pd.ExcelWriter('efficient_exhaustive.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

# ALTERNATING TESTS:
# ARBITRARY POLYPHASE TESTS
# These tests, for lengths 8-20, aim to find near-alternating codes with the number of codes ranging from 4-10
# For binary codes, we only consider an even number of codes, and compare random and hill climbing algorithms
# For polyphase codes, we consider any number of codes, and compare random and great deluge algorithms.

def random_alternating_binary_test():
    print('random alternating binary')
    length = []
    number_of_codes = []
    random_codes = []
    random_values = []
    for i in range(8, 20):
        print(i)
        for j in range(4, 10, 2):
            print(j)
            length.append(i)
            number_of_codes.append(j)
            x = random_alternating_binary(i, j)
            random_codes.append(x[0])
            random_values.append(x[1])
            print(x[1])

    df = pd.DataFrame({'Length': length, 'Number of codes': number_of_codes,
                       'Code': random_codes, 'PSL Value': random_values})
    writer = pd.ExcelWriter('random_alternating_binary.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


def random_alternating_polyphase_test():
    print('random alternating polyphase')
    length = []
    number_of_codes = []
    random_codes = []
    random_values = []
    for i in range(8, 20):
        print(i)
        for j in range(4, 10):
            print(j)
            length.append(i)
            number_of_codes.append(j)
            x = random_alternating_polyphase(i, j)
            random_codes.append(x[0])
            random_values.append(x[1])
            print(x[1])

    df = pd.DataFrame({'Length': length, 'Number of codes': number_of_codes,
                       'Code': random_codes, 'PSL Value': random_values})
    writer = pd.ExcelWriter('random_alternating_polyphase.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


def hill_climbing_alternating():
    print('hill_climbing_alternating')
    length = []
    number_of_codes = []
    hill_climbing_codes = []
    hill_climbing_values = []
    for i in range(8, 20):
        print(i)
        for j in range(4, 10, 2):
            print(j)
            length.append(i)
            number_of_codes.append(j)
            x = hill_climbing_iterative_alternating(i, j, sdls_local_search)
            print('hi')
            hill_climbing_codes.append(x[0])
            hill_climbing_values.append(x[1])

    df = pd.DataFrame({'Length': length, 'Number of codes': number_of_codes,
                       'Code': hill_climbing_codes, 'PSL Value': hill_climbing_values})
    writer = pd.ExcelWriter('hill_climb_alternating.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def great_deluge_alternating_test():
    print('great_deluge_alternating')
    length = []
    number_of_codes = []
    great_deluge_codes = []
    great_deluge_values = []
    for i in range(8, 20):
        print(i)
        for j in range(4, 10):
            print(j)
            length.append(i)
            number_of_codes.append(j)
            x = great_deluge_handler(i, great_deluge_alternating, j)
            great_deluge_codes.append(x[0])
            great_deluge_values.append(x[1])

    df = pd.DataFrame({'Length': length, 'Number of codes': number_of_codes,
                       'Code': great_deluge_codes, 'PSL Value': great_deluge_values})
    writer = pd.ExcelWriter('great_deluge_alternating.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

"""
def gen_scatter_plot_sidelobes(num_elem, N, op_on_list, gen_codes):
    x = []
    y = []
    for i in range(4, 5):
        x.append(i)
        val = peak_sidelobes_general(i, num_elem, N, op_on_list, gen_codes)
        y.append(val[1])
    plt.scatter(x, y)
    plt.xlabel('Length of code')
    plt.ylabel('Minimum Peak Sidelobe')
    plt.title('Minimum Peak Doppler Sidelobes')
    plt.axis([4, 15, 0, 150])
    plt.grid(True)
    plt.show()


def gen_plot_sidelobes(num_elem, op_on_list, gen_codes):
    x = []
    y = []
    for i in range(2, 18):
        x.append(i)
        val = peak_sidelobes_general(i, num_elem, i, op_on_list, gen_codes)
        y.append(val[1])

    plt.scatter(x, y)
    plt.xlabel('Length of code')
    plt.ylabel('Minimum Peak Sidelobe')
    plt.title('Minimum Peak Doppler Sidelobes')
    plt.axis([0, 20, 0, 150])
    plt.grid(True)
    plt.show()
"""
