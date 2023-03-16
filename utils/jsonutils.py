import csv, json

def _compose_records(epoch, data):
    tot_labels = ['epoch']
    tot_vaccs  = ['{} (acc.)'.format(epoch)]
    tot_vloss  = ['{} (loss)'.format(epoch)]

    # loop over the data
    for each_bits, (each_vacc, each_vloss) in data.items():
        tot_labels.append('{}-bits'.format(each_bits))
        tot_vaccs.append('{:.4f}'.format(each_vacc))
        tot_vloss.append('{:.4f}'.format(each_vloss))

    # return them
    return tot_labels, tot_vaccs, tot_vloss


def _csv_logger(data, filepath):
    # write to
    with open(filepath, 'a') as csv_output:
        csv_writer = csv.writer(csv_output)
        csv_writer.writerow(data)
    # done.

def _store_prefix(parameters):
    prefix = ''

    # store the attack info.
    prefix += 'attack_{}_{}_{}_w{}_a{}-'.format( \
        ''.join([str(each) for each in parameters['attack']['numbit']]),
        parameters['attack']['lratio'],
        parameters['attack']['margin'],
        ''.join([each[0] for each in parameters['model']['w-qmode'].split('_')]),
        ''.join([each[0] for each in parameters['model']['a-qmode'].split('_')]))

    # optimizer info
    prefix += 'optimize_{}_{}_{}'.format( \
            parameters['params']['epoch'],
            parameters['model']['optimizer'],
            parameters['params']['lr'])
    return prefix