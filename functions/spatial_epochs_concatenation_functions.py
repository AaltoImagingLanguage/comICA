#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np 
import mne

def generate_events(epochs_number, event_ids):
    events = []

    sample_number = 0
    for condition in epochs_number:
        print("Events condition is", condition,
              "and number of epochs is", epochs_number[condition])
        for number in range(epochs_number[condition]):
            events.append([sample_number*210, 0, event_ids[condition]])
            sample_number += 1

    return np.array(events)


def create_info(rec_1_epochs, rec_2_epochs):
    rec_1_sensors = rec_1_epochs.ch_names
    rec_2_sensors = rec_2_epochs.ch_names
    sfreq = rec_2_epochs.info['sfreq']

    print("Filling in the info structure")
    ch_types_rec_1 = [mne.channel_type(rec_1_epochs.info, ch_idx)
                    for ch_idx in range(len(rec_1_sensors))]
    ch_types_rec_2 = [mne.channel_type(
        rec_2_epochs.info, ch_idx) for ch_idx in range(len(rec_2_sensors))]

    info = mne.create_info(ch_names=rec_1_epochs.ch_names + rec_2_epochs.ch_names,
                           sfreq=sfreq, ch_types=ch_types_rec_1+ch_types_rec_2)
    return info

def combine_epochs(event_ids, 
                   rec_1_epochs, 
                   rec_2_epochs, 
                   tmin, tmax):

    counts_rec_1 = {k: sum(rec_1_epochs.events[:, 2] == rec_1_epochs.event_id[k])
                  for k in event_ids}

    counts_rec_2 = {k: sum(rec_2_epochs.events[:, 2] == rec_2_epochs.event_id[k])
                   for k in event_ids}

    epochs_number = {k: min(count1, count2) for (
        k, count1), (_, count2) in zip(counts_rec_1.items(),
                                       counts_rec_2.items())}

    rec_1_conditions = []
    rec_2_conditions = []

    for event in event_ids:
        if epochs_number[event] == 0:
            print("No events for condition", event)
        else:
            rec_1_conditions.append(rec_1_epochs[event][:epochs_number[event]])
            rec_2_conditions.append(rec_2_epochs[event][:epochs_number[event]])
    rec_1_epochs2 = mne.concatenate_epochs(rec_1_conditions)
    rec_2_epochs2 = mne.concatenate_epochs(rec_2_conditions)

    data = np.concatenate(
        (rec_1_epochs2.get_data()[:, :, :],
         rec_2_epochs2.get_data()[:, :, :]),
        axis=1)

    events = rec_1_epochs2.events
    info = create_info(rec_1_epochs, rec_2_epochs)

    return mne.EpochsArray(data, info=info,
                           events=events, event_id=event_ids,
                           tmin=tmin)

    return mne.EpochsArray(data, info=info,  events=events,
                           event_id=event_ids, tmin=tmin)



