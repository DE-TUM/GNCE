"""
    Title: Deep Unsupervised Cardinality Estimation Source Code
    Author:  Amog Kamsetty, Chenggang Wu, Eric Liang, Zongheng Yang
    Date: 2020
    Availability: https://github.com/naru-project/naru

    Source Code used as is or modified from the above mentioned source
"""

import argparse
import collections
import glob
import os
import re

import numpy as np
import pandas as pd
import torch

import common
import datasets
import estimators as estimators_lib
import made

# For inference speed.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

DEVICE = 'cpu'
print('Device', DEVICE)

parser = argparse.ArgumentParser()

parser.add_argument('--inference-opts',
                    action='store_true',
                    help='Tracing optimization for better latency.')

parser.add_argument('--dataset', type=str, default='swdf_star_2', help='Dataset.')
parser.add_argument('--err-csv',
                    type=str,
                    default='results.csv',
                    help='Save result csv to what path?')
parser.add_argument('--glob',
                    type=str,
                    help='Checkpoints to glob under models/.')
parser.add_argument('--blacklist',
                    type=str,
                    help='Remove some globbed checkpoint files.')
parser.add_argument(
    '--column-masking',
    action='store_true',
    help='Turn on wildcard skipping.  Requires checkpoints be trained with '\
    'column masking.')

# MADE.
parser.add_argument('--fc-hiddens',
                    type=int,
                    default=512,
                    help='Hidden units in FC.')
parser.add_argument('--layers', type=int, default=3, help='# layers in FC.')
parser.add_argument('--residual', action='store_true', help='ResMade?')
parser.add_argument('--direct-io', action='store_true', help='Do direct IO?')

parser.add_argument(
    '--input-encoding',
    type=str,
    default='binary',
    help='Input encoding for MADE/ResMADE, {binary, one_hot, embed}.')
parser.add_argument(
    '--output-encoding',
    type=str,
    default='one_hot',
    help='Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, '
    'then input encoding should be set to embed as well.')

parser.add_argument('--query-type', type=str, default='star', help='The query type.')

args = parser.parse_args()

def MakeTable():
    assert args.dataset in ['swdf_star_2', 'swdf_chain_2']
    if args.dataset == 'swdf_chain_2':
        table = datasets.LoadChain2('swdf_chain_2.csv')
    elif args.dataset == 'swdf_star_2':
        table = datasets.LoadStar2('swdf_star_2.csv')

    oracle_est = estimators_lib.Oracle(table)
    print('make table oracle estimate')
    print(oracle_est)

    return table, None, oracle_est


def ErrorMetric(est_card, card):
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)

def Query(estimators,
          do_print=False,
          oracle_card=None,
          query=None,
          table=None,
          oracle_est=None,
          true_cardinality=0,
          query_id=-1):
    assert query is not None
    cols, ops, vals, discretized_vals = query

    ### Actually estimate the query.

    def pprint(*args, **kwargs):
        if do_print:
            print(*args, **kwargs)

    card = true_cardinality

    if card == 0:
        print('inside return')
        return

    for est in estimators:
        est_card = est.Query(cols, ops, vals, discretized_vals)
        err = ErrorMetric(est_card, card)
        est.AddError(err, est_card, card)


def ReportEsts(estimators):
    v = -1
    for est in estimators:
        print(est.name, 'max', np.max(est.errs), '99th',
              np.quantile(est.errs, 0.99), '95th', np.quantile(est.errs, 0.95),
              'median', np.quantile(est.errs, 0.5))
        v = max(v, np.max(est.errs))
    return v


def RunN(table,
         cols,
         estimators,
         rng=None,
         log_every=50,
         num_filters=11,
         oracle_cards=None,
         oracle_est=None,
         needed_train_data=None,
         query_type='star'):
    '''
     Read existing queries and compress, change the path to the queries
    :param table:
    :param cols:
    :param estimators:
    :param rng:
    :param num:
    :param log_every:
    :param num_filters:
    :param oracle_cards:
    :param oracle_est:
    :param needed_train_data:
    :param query_type:
    :return:
    '''
    log_every = 500

    # the element which was used for compressing the columns
    compressor_elem = table.compressor_element

    queries = list()
    discretized_queries = list()
    columns_final = list()
    operators_final = list()
    true_cardinalities = list()
    actual_query_lines = list()

    with open('queries/eval_swdf_{}_2_0.txt'.format(query_type)) as f_tmp:
        for num_lin, line in enumerate(f_tmp):

            if query_type == 'star':
                split_line = line.split(':')
            elif query_type == 'chain':
                split_line = line.split(',')
            else:
                print('pick star or chain')
                exit(1)

            query_parts = split_line[0].strip().replace(',','-').split('-')
            true_card = int(split_line[1].strip())

            true_cardinalities.append(true_card)

            final_columns_for_query = list()
            final_column_values = list()
            # cannot take 'i' since it doesn't correspond to the actual number of columns that are present after the modification
            modified_columns_index = 0
            for i, query_part in enumerate(query_parts):
                # only take the columns that are bound (not *)
                if query_part.strip() is not '*':
                    # if the column is part of the split columns then it should be sent to the compressor
                    if i in compressor_elem.split_columns_index:
                        # every column at the beginning will be split into 2 columns
                        how_many_times_compressed = 2
                        quotient, reminder = compressor_elem.split_single_value_for_column(int(query_part.strip()), i)
                        # save the reminder for the future
                        all_reminders = list()
                        all_reminders.append(reminder)

                        # split the column into the required number of columns
                        while how_many_times_compressed < compressor_elem.root:
                            # get the quotient and reminder from the quotient in the previous iteration
                            quotient, reminder = compressor_elem.split_single_value_for_column(int(quotient), i)

                            # save the reminder, it will represent a separate column
                            all_reminders.append(reminder)
                            how_many_times_compressed += 1

                        # store the information for the quotient as a separate column
                        final_columns_for_query.append(cols[modified_columns_index])
                        modified_columns_index += 1
                        final_column_values.append(int(quotient))

                        # iterate over the reminders in a reversed order such that the last reminder
                        # is actually the reminder for the quotient the one after that is the reminder for the number
                        # made by the quotient and the first reminder, etc...
                        for num_rem, rem_val in enumerate(reversed(all_reminders)):
                            # store the id of the column
                            final_columns_for_query.append(cols[modified_columns_index])
                            if num_rem + 1 < len(all_reminders): # do not increment for the last column, this is done outside of the if/else statement
                                modified_columns_index += 1
                            # store the value of the column
                            final_column_values.append(int(rem_val))
                    else:
                        # for the columns that are not split
                        final_columns_for_query.append(cols[modified_columns_index])
                        final_column_values.append(int(query_part.strip()))
                else:
                    # needed so that we keep the correct column index
                    if i in compressor_elem.split_columns_index:
                        modified_columns_index += compressor_elem.root - 1

                # go to the next modified column
                modified_columns_index += 1

            # create the actual query by taking the discretized values from the column
            new_column_values = list()
            for column_id, column in enumerate(final_columns_for_query):
                new_column_values.append(common.Discretize(column, [final_column_values[column_id]])[0])

            columns_final.append(np.array(final_columns_for_query))
            operators_final.append(np.array(['='] * int(len(final_columns_for_query))))
            queries.append(final_column_values)
            discretized_queries.append(new_column_values)


            # create the actual query
            query = np.array(final_columns_for_query), np.array(['='] * int(len(final_columns_for_query))), np.array(
                final_column_values), np.array(new_column_values)

            # call the query estimator
            Query(estimators,
                  False,  # do not print anything
                  oracle_card=oracle_cards[i] if oracle_cards is not None and i < len(oracle_cards) else None,
                  query=query,#custom_query
                  table=table,
                  oracle_est=oracle_est,
                  true_cardinality=true_card)

    return False

def MakeMade(scale, cols_to_train, seed, fixed_ordering=None):
    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        args.layers if args.layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
    ).to(DEVICE)

    return model

def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


def SaveEstimators(path, estimators, return_df=False):
    # name, query_dur_ms, errs, est_cards, true_cards
    results = pd.DataFrame()

    for est in estimators:
        print(len([est.name] * len(est.errs)))
        print(len(est.errs))
        print(len(est.est_cards))
        print(len(est.true_cards))
        print(len(est.query_dur_ms))
        data = {
            'est': [est.name] * len(est.errs),
            'err': est.errs,
            'est_card': est.est_cards,
            'true_card': est.true_cards,
            'query_dur_ms': est.query_dur_ms,
        }
        results = results.append(pd.DataFrame(data))
    if return_df:
        return results
    results.to_csv(path, index=False)


def LoadOracleCardinalities():
    ORACLE_CARD_FILES = {

    }
    path = ORACLE_CARD_FILES.get(args.dataset, None)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        assert len(df) == 2000, len(df)
        return df.values.reshape(-1)
    return None


def Main():
    all_ckpts = glob.glob('./models/{}'.format(args.glob))
    if args.blacklist:
        all_ckpts = [ckpt for ckpt in all_ckpts if args.blacklist not in ckpt]

    selected_ckpts = all_ckpts
    oracle_cards = LoadOracleCardinalities()
    print('ckpts', selected_ckpts)

    # OK to load tables now
    table, train_data, oracle_est = MakeTable()

    cols_to_train = table.columns
    table.Name()

    Ckpt = collections.namedtuple(
        'Ckpt', 'epoch model_bits bits_gap path loaded_model seed')
    parsed_ckpts = []

    for s in selected_ckpts:
        z = re.match('.+model([\d\.]+)-data([\d\.]+).+seed([\d\.]+).*.pt',
                     s)

        assert z
        model_bits = float(z.group(1))
        data_bits = float(z.group(2))
        seed = int(z.group(3))
        bits_gap = model_bits - data_bits

        order = None

        if args.dataset in ['swdf_star_2', 'swdf_chain_2']:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=order,
            )
        else:
            assert False, args.dataset

        assert order is None or len(order) == model.nin, order
        ReportModel(model)
        print('Loading ckpt:', s)

        model.load_state_dict(torch.load(s), strict=False)
        # when needed to load the model on CPU
        # model.load_state_dict(torch.load(s, map_location=torch.device('cpu')), strict=False)
        model.eval()

        print(s, bits_gap, seed)

        parsed_ckpts.append(
            Ckpt(path=s,
                 epoch=None,
                 model_bits=model_bits,
                 bits_gap=bits_gap,
                 loaded_model=model,
                 seed=seed))

    # Estimators to run.
    estimators = [
        estimators_lib.BaseDistributionEstimation(c.loaded_model,
                                   table,
                                   1,
                                   device=DEVICE,
                                   shortcircuit=args.column_masking)
        for c in parsed_ckpts
    ]
    for est, ckpt in zip(estimators, parsed_ckpts):
        est.name = str(est) + '_{}_{:.3f}'.format(ckpt.seed, ckpt.bits_gap)

    if args.inference_opts:
        print('Tracing forward_with_encoded_input()...')
        for est in estimators:
            encoded_input = est.model.EncodeInput(
                torch.zeros(1, est.model.nin, device=DEVICE))

            # NOTE: this line works with torch 1.0.1.post2 (but not 1.2).
            # The 1.2 version changes the API to
            # torch.jit.script(est.model) and requires an annotation --
            # which was found to be slower.
            est.traced_fwd = torch.jit.trace(
                est.model.forward_with_encoded_input, encoded_input)

    if len(estimators):
        print('create estimators')
        RunN(table,
             cols_to_train,
             estimators,
             rng=np.random.RandomState(1234),
             log_every=1,
             num_filters=None,
             oracle_cards=oracle_cards,
             oracle_est=oracle_est,
             needed_train_data=common.TableDataset(table)[:,:],
             query_type=args.query_type)

    SaveEstimators(args.err_csv, estimators)
    print('...Done, result:', args.err_csv)


if __name__ == '__main__':
    print('Main called first')
    Main()
