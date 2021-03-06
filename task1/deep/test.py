import argparse
import torch
import pickle
import codecs

from evaluate import predict
from data import *

EOS_symbol = "<EOS>"
EOS_index = 0
PAD_index = 1
PAD_symbol = "#"
UNK_index = 2
UNK_symbol='@'

def replace_UNK(input_text, input_encoded, pred):
    updated_pred = []
    if UNK_symbol in pred:
        print("UNKNOWN: %s:" % pred)
        in_copy = [i for i, c in enumerate(input_encoded)\
                   if c==UNK_symbol]

        copy_count = 0
        for p in pred:
            if p == UNK_symbol:
                # Replace the unknown with the corresponding
                # char in the input sequence.
                try:
                    """
                    Try to use the corresponding copy symbol.
                    """
                    updated_pred.append(input_text[in_copy[copy_count]])
                    copy_count+=1
                except:
                    """
                    Probably would make sense to just append most
                    frequent char here. Just randomly choose first char
                    of the input for now...
                    """
                    updated_pred.append(input_text[0])
            else:
                updated_pred.append(p)

        return ''.join(updated_pred)
    else:
        return pred

def make_predictions(pairs, encoder, decoder, char2i, outputfn,\
                     batch_size=100, use_cuda=True):
    i2char = {c: i for i, c in char2i.items()}
    batches = get_batches(pairs, batch_size, char2i,\
                          PAD_symbol, use_cuda, test_mode=True)
    output_strings = []
    out = open(outputfn, "w")
    for batch in batches:
        preds = predict(encoder, decoder, batch,\
                    list(char2i.keys()), use_cuda)

        for j in range(batch.size):
            eos = (preds[:, j] == EOS_index).\
                  nonzero().data[1][0]
            # Write the prediction up to the
            # second eos
            #print(''.join([i2char[int(c)] for c in\
            #               preds[1:eos, j]]))
            input_text = batch.inputs[j]
            input_enc = [i2char[int(c)] for c in \
                         batch.input_variable.t()[j]]
            pred = ''.join([i2char[int(c)] for c in\
                     preds[1:eos, j]])
            pred = replace_UNK(input_text, input_enc, pred)

            out.write(pred)
            out.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test")
    parser.add_argument('testfn', metavar='testfn',\
                        help='filename of test file')
    parser.add_argument('output_dir', metavar='output_dir',\
                        help='directory to write predictions to')
    parser.add_argument('encoder', metavar='encoder', \
                        help='The file for the encoder')
    parser.add_argument('decoder', metavar='decoder', \
                        help='The file for the decoder')
    parser.add_argument('char2i', metavar='char2i', \
                        help='Pickle file for the mapping')
    parser.add_argument('lang', metavar='lang',\
                        help='The language')
    parser.add_argument('setting', metavar='setting',\
                        help='low/medium/high')
    parser.add_argument('batch_size', help=\
                        'The suize of each Batch')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    lang = args.lang
    setting = args.setting
    test_data = Data(args.testfn)
    encoder = torch.load(args.encoder)
    decoder = torch.load(args.decoder)
    char2i = pickle.load(open(args.char2i, "rb"))
    output_dir = args.output_dir
    output = "%s/%s-preds-%s" % (output_dir, lang, setting)
    batch_size = int(args.batch_size)
    use_cuda = args.gpu

    print(char2i)
    if use_cuda:
        enocder = encoder.cuda()
        decoder = decoder.cuda()
    
    test_pairs = [([EOS_symbol] + i + [EOS_symbol],\
                  [EOS_symbol] + o + [EOS_symbol])\
                  for i, o in test_data.pairs]
    make_predictions(test_pairs, encoder, decoder, char2i, output,\
                     batch_size, use_cuda)
