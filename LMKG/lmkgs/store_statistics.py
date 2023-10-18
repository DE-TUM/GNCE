import math
import numpy as np

def check_network_estimates(predicted_cardinality, true_cardinality, statistics_file_name, print_time, encoding_avg_time = 0, avg_time_prediction_ms = 0, store_file_name = "name", nb_outliers = 0):
    correct_estimates = 0
    wrong_estimates = 0
    max_correct = 0
    correct_estimates_not_one = 0
    wrong_estimates_not_one = 0
    average_estimation_error = 0
    all_wrong_estimates = list()
    mse_for_all_results = 0
    q_loss_for_all_results = 0
    mae_for_all_results = 0
    errors_per_bucket = dict()
    greater_10 = 0
    greater_100 = 0
    greater_1000 = 0
    q_error_per_range = dict()
    all_q_errors = list()
    max_nb = 0
    max_error = 0

    if nb_outliers > 0:
        store_file_name += "_" + str(nb_outliers) + "outliers"

    with open('statistics/'+store_file_name+'.txt','w') as store_file:
        store_file.write('id,est_card,true_card,enc_avg_time,pred_avg_time,total_avg_time\n')
        for i in range(len(predicted_cardinality)):
            pred_card = float(predicted_cardinality[i][0])
            true_card = float(true_cardinality[i])

            store_file.write(str(i)+','+str(pred_card) +","+str(true_card)+','+str(encoding_avg_time)+','+str(avg_time_prediction_ms)+','+str(avg_time_prediction_ms+encoding_avg_time)+'\n')
            store_file.flush()
    print('creation of statistics file done')

    print('number of results %d ' % len(predicted_cardinality))
    for i in range(len(predicted_cardinality)):
        pred_card = float(predicted_cardinality[i][0])
        # predicted_cardinality[i][0] = float(predicted_cardinality[i][0])
        # calculating MSE
        mse_tmp = pow((pred_card - true_cardinality[i]), 2.0)
        mse_for_all_results += mse_tmp

        # calculate MAE
        mae_tmp = abs((pred_card - true_cardinality[i]))
        mae_for_all_results += mae_tmp

        # calculate Q_ERROR
        tmp_card_q_error_changed = (pred_card + 1) if pred_card == 0.0 else pred_card
        q_er_true_card = true_cardinality[i]
        if pred_card == 0:
            q_er_true_card = q_er_true_card + 1
        q_err_tmp = max(tmp_card_q_error_changed, q_er_true_card) / min(tmp_card_q_error_changed, q_er_true_card)
        q_loss_for_all_results += q_err_tmp
        all_q_errors.append(q_err_tmp)

        # adding an error per a specific bucket
        # doing a logarithm with base 3 as a has function
        position_dictionary = math.floor(math.log(int(true_cardinality[i]), 3))
        existing_errors_for_bucket = errors_per_bucket.get(position_dictionary)
        # check if the dictionary already has this bucket
        if existing_errors_for_bucket is None:
            existing_errors_for_bucket = list()
        # store the errors for the bucket
        existing_errors_for_bucket.append(abs(int(round(pred_card)) - int(true_cardinality[i])))
        # update the dictionary
        errors_per_bucket[position_dictionary] = existing_errors_for_bucket

        if true_cardinality[i] > 10:
            greater_10 += 1
        if true_cardinality[i] > 100:
            greater_100 += 1
        if true_cardinality[i] > 1000:
            greater_1000 += 1

        # print(predicted_cardinality[i])
        if int(round(pred_card)) == int(true_cardinality[i]):
            if true_cardinality[i] > max_correct:
                max_correct = true_cardinality[i]
            correct_estimates += 1
            if true_cardinality[i] != 1:
                correct_estimates_not_one += 1
        else:
            print('true %d - est %d ' % (int(true_cardinality[i]), int(pred_card)))
            all_wrong_estimates.append(abs(int(true_cardinality[i]) - int(round(pred_card))))
            # print('wrong estimate ->  true: %d - estimate: %d - estimate_real: %.2f' % (true_cardinality[i],round(float(predicted_cardinality[i][0])), float(predicted_cardinality[i][0])))
            wrong_estimates += 1
            if true_cardinality[i] != 1:
                wrong_estimates_not_one += 1
            average_estimation_error += abs(true_cardinality[i] - pred_card)

            differ = abs(int(true_cardinality[i]) - int(round(pred_card)))
            if differ > max_error:
                max_error = differ
                max_nb = int(true_cardinality[i])



        if position_dictionary > 7:
            continue
            # position_dictionary = 7
        existing_q_error = q_error_per_range.get(position_dictionary)
        if existing_q_error is None:
            existing_q_error = list()
        existing_q_error.append(q_err_tmp)
        q_error_per_range[position_dictionary] = existing_q_error
    print()
    average_estimation_error = (average_estimation_error * 1.0) / wrong_estimates
    all_wrong_estimates.sort(reverse=True)
    final_print_statistics = ''
    final_print_statistics += ('maximal corrected estimate: %d \n' % max_correct)
    final_print_statistics += ('correct estimates that are not 1: %d \n' % correct_estimates_not_one)
    final_print_statistics += ('wrong estimates that are not 1: %d \n' % wrong_estimates_not_one)
    final_print_statistics += ('correct estimates: %d \n' % correct_estimates)
    final_print_statistics += ('wrong estimates: %d \n' % wrong_estimates)
    final_print_statistics += ('total number of data: %d \n' % len(predicted_cardinality))

    median_q_err = np.median(all_q_errors)

    # print(all_wrong_estimates)
    # print()
    final_print_statistics += ('final errors:\n')
    final_print_statistics += ("Mean Squared Error: %.3f\n" % (mse_for_all_results / len(true_cardinality)))
    final_print_statistics += ("Mean Absolute Error: %.3f\n" % (mae_for_all_results / len(true_cardinality)))
    final_print_statistics += ("Q-Error: %.3f\n" % (q_loss_for_all_results / len(true_cardinality)))
    final_print_statistics += ("Median-Q-Error: %.3f\n" % (np.median(all_q_errors)))
    final_print_statistics += ('average estimation error: %.3f\n' % average_estimation_error)
    final_print_statistics += ('median estimation error: %.3f\n' % np.median(all_wrong_estimates))
    final_print_statistics += ('maximal estimation error: %d\n' % all_wrong_estimates[0])
    final_print_statistics += ('maximal q error: prediction %d true %d q-err %.3f\n' % (max_error, max_nb, max(max_error, max_nb) / min(max_error, max_nb)))
    final_print_statistics += ('minimal estimation error: %d\n' % all_wrong_estimates[len(all_wrong_estimates) - 1])
    final_print_statistics += ('25th percentile: %.3f\n' % np.percentile(all_wrong_estimates, 25))
    final_print_statistics += ('50th percentile: %.3f\n' % np.percentile(all_wrong_estimates, 50))
    final_print_statistics += ('75th percentile: %.3f\n' % np.percentile(all_wrong_estimates, 75))
    final_print_statistics += ('95th percentile: %.3f\n' % np.percentile(all_wrong_estimates, 95))
    final_print_statistics += ('99th percentile: %.3f\n' % np.percentile(all_wrong_estimates, 99))
    final_print_statistics += ('greater than 10 %d\n' % greater_10)
    final_print_statistics += ('greater than 100 %d\n' % greater_100)
    final_print_statistics += ('greater than 1000 %d\n' % greater_1000)

    final_q_errors_per_range = dict()
    median_q_errors_per_range = dict()
    # print the errors for a specific range of true cardinalities
    for key in errors_per_bucket:
        values_per_key = errors_per_bucket[key]
        sum_errors = 0
        for val in values_per_key:
            sum_errors += val
        start_bucket = np.power(3, key)
        end_bucket = np.power(3, (key + 1))
        if key <= 7:
            # if True:
            range_key = '$[3^{' + str(key) + '},3^{' + str(key + 1) + '})$'
            sum_all_vals = sum(val for val in q_error_per_range[key])
            final_q_errors_per_range[range_key] = sum_all_vals / len(q_error_per_range[key])

            existing_median_val = median_q_errors_per_range.get(range_key)
            if existing_median_val is None:
                existing_median_val = list()
            existing_median_val.append(np.median(q_error_per_range[key]))
            median_q_errors_per_range[range_key] = existing_median_val

            final_print_statistics += (
                        'q-err-range:' + range_key + ': ' + str(sum_all_vals / len(q_error_per_range[key])) + '\n')
            final_print_statistics += (
                        'median-q-err-range:' + range_key + ': ' + str(np.median(q_error_per_range[key])) + '\n')
        final_print_statistics += ('average error for bucket %d is %.3f [%d, %d) - number of points in bucket %d\n' % (
        key, (sum_errors / len(values_per_key)), start_bucket, end_bucket, len(values_per_key)))
    print(final_print_statistics)
    # create_csv_statistics_file(final_print_statistics, statistics_file_name, print_time)