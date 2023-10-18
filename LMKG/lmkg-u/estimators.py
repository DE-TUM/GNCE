"""
    Title: Deep Unsupervised Cardinality Estimation Source Code
    Author:  Amog Kamsetty, Chenggang Wu, Eric Liang, Zongheng Yang
    Date: 2020
    Availability: https://github.com/naru-project/naru

    Source Code used as is or modified from the above mentioned source
"""

import time
import numpy as np
import torch
import made

# only equality (=) is needed
OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal
}


class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.name = 'CardEst'

    def Query(self, columns, operators, vals):

        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['=', '='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self):
        self.query_starts.append(time.time())

    def OnEnd(self):
        self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)

    def AddError(self, err):
        self.errs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts, self.query_dur_ms, self.errs, self.est_cards,
            self.true_cards
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        print(est.name, "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "time_ms",
              np.mean(est.query_dur_ms))

def FillInUnqueriedColumnsAndDiscretized(table, columns, operators, vals, discretized_vals):
    """Allows for some terms to be unqueried (i.e., wildcard).

    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural term order.

    A None in ops/vals means that term slot is unqueried.
    """
    ncols = len(table.columns)
    cs = table.columns
    os, vs, dsv = [None] * ncols, [None] * ncols, [None] * ncols

    for c, o, v, d in zip(columns, operators, vals, discretized_vals):
        idx = table.ColumnIndex(c.name)
        os[idx] = o
        vs[idx] = v
        dsv[idx] = d

    return cs, os, vs, dsv

class BaseDistributionEstimation(CardEst):
    '''
        Distribution estimation based on the AR model.
    '''
    def __init__(
            self,
            model,
            table,
            r,
            device=None,
            seed=False,
            cardinality=None,
            shortcircuit=False  # Skip sampling on wildcards?
    ):
        super(BaseDistributionEstimation, self).__init__()
        torch.set_grad_enabled(False)
        self.model = model
        self.table = table
        self.shortcircuit = shortcircuit

        self.num_samples = 1

        self.seed = seed
        self.device = device

        self.cardinality = cardinality
        if cardinality is None:
            self.cardinality = table.cardinality

        with torch.no_grad():
            self.init_logits = self.model(
                torch.zeros(1, self.model.nin, device=device))

        self.dom_sizes = [c.DistributionSize() for c in self.table.columns]
        self.dom_sizes = np.cumsum(self.dom_sizes)

        # Inference optimizations below.
        self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        self.traced_encode_input = model.EncodeInput

        if 'MADE' in str(model):
            for layer in model.net:
                if type(layer) == made.MaskedLinear:
                    if layer.masked_weight is None:
                        layer.masked_weight = layer.mask * layer.weight
                        print('Setting masked_weight in MADE, do not retrain!')

        for p in model.parameters():
            p.detach_()
            p.requires_grad = False
        self.init_logits.detach_()

        #self.inp represents the encoding for the input
        with torch.no_grad():
            self.kZeros = torch.zeros(self.num_samples,
                                      self.model.nin,
                                      device=self.device)

            self.inp = self.traced_encode_input(self.kZeros)

            self.inp = self.inp.view(self.num_samples, -1)

    def __str__(self):
        if self.num_samples:
            n = self.num_samples
        else:
            n = int(self.r * self.table.columns[0].DistributionSize())
        return 'psample_{}'.format(n)

    def _sample_n(self,
                  num_samples,
                  ordering,
                  columns,
                  operators,
                  vals,
                  discretized_vals,
                  inp=None, doPartial=False):
        ncols = len(columns)


        # only take one example
        inp = self.inp[:1]

        original_vals = vals.copy()

        '''
            Do the encoding for the wildcards 
        '''
        for i in range(ncols):
            natural_idx = i if ordering is None else ordering[i]
            if operators[natural_idx] is None:
                '''encoding for the wildcards'''
                if natural_idx == 0:
                    self.model.EncodeInput(
                        None,
                        natural_col=0,
                        out=inp[:, :self.model.
                            input_bins_encoded_cumsum[0]])
                else:
                    l = self.model.input_bins_encoded_cumsum[natural_idx -
                                                             1]
                    r = self.model.input_bins_encoded_cumsum[natural_idx]
                    self.model.EncodeInput(None,
                                           natural_col=natural_idx,
                                           out=inp[:, l:r])
            else:
                # put them in the required format
                data_to_encode = torch.LongTensor([discretized_vals[natural_idx]])#.view(-1, 1)

                if natural_idx == 0:
                    self.model.EncodeInput(
                        data_to_encode,
                        natural_col=0,
                        out=inp[:, :self.model.
                            input_bins_encoded_cumsum[0]])

                else:
                    # starting bit postion for current column
                    l = self.model.input_bins_encoded_cumsum[natural_idx - 1]
                    # ending bit postion for current column
                    r = self.model.input_bins_encoded_cumsum[natural_idx]

                    self.model.EncodeInput(data_to_encode,
                                           natural_col=natural_idx,
                                           out=inp[:, l:r])

        # create the logits for the encoded query
        logits = self.model.forward_with_encoded_input(inp)

        p_x_1 = 1.0
        for i in range(ncols):
            natural_idx = i if ordering is None else ordering[i]

            probs_i = torch.softmax(
                self.model.logits_for_col(natural_idx, logits), 1)

            if operators[natural_idx] is not None:
                # the set terms
                idx = np.where(columns[natural_idx].all_distinct_values == original_vals[natural_idx])[0]

                p_x_1 *= probs_i[0][idx].item()
            else:
                p_x_1 *= probs_i[0].sum().item()

        return p_x_1

    def Query(self, columns, operators, vals, discretized_vals):
        '''fill the unqueried terms'''
        columns, operators, vals, discretized_vals = FillInUnqueriedColumnsAndDiscretized(
            self.table, columns, operators, vals, discretized_vals)

        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(len(columns))
            orderings = [ordering]

        num_orderings = len(orderings)

        # order idx (first/second/... to be sample) -> x_{natural_idx}.
        inv_ordering = [None] * len(columns)
        for natural_idx in range(len(columns)):
            inv_ordering[ordering[natural_idx]] = natural_idx

        with torch.no_grad():
            inp_buf = self.inp.zero_()
            # Fast (?) path.
            if num_orderings == 1:
                ordering = orderings[0]
                self.OnStart()
                p = self._sample_n(
                    self.num_samples,
                    inv_ordering,
                    columns,
                    operators,
                    vals,
                    discretized_vals,
                    inp=inp_buf, doPartial=False)
                self.OnEnd()

                result = np.ceil(p * self.cardinality).astype(dtype=np.int32,
                                                               copy=False)
                print('result %.3f' % (result if result > 0 else 1))

        return result if result > 0 else 1

class Oracle(CardEst):
    """Returns true cardinalities."""

    def __init__(self, table, limit_first_n=None):
        super(Oracle, self).__init__()
        self.table = table
        self.limit_first_n = limit_first_n

    def __str__(self):
        return 'oracle-est: ' + str(self.limit_first_n)

    def Query(self, columns, operators, vals, return_masks=False):
        assert len(columns) == len(operators) == len(vals)
        self.OnStart()

        bools = None
        for c, o, v in zip(columns, operators, vals):
            if self.limit_first_n is None:
                inds = OPS[o](c.data, v)
            else:
                # For data shifts experiment.
                inds = OPS[o](c.data[:self.limit_first_n], v)

            if bools is None:
                bools = inds
            else:
                bools &= inds
        c = bools.sum()
        self.OnEnd()
        if return_masks:
            return bools
        return c