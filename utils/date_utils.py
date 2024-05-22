from datetime import datetime, timedelta
import numpy as np


def get_chunked_date_range(
    startdate, enddate, num_chunks, max_chunk_len=None, min_chunk_len=None
):
    """Split a date range into chunks of approximately the same length.

    Parameters
    ----------
    startdate : datetime.datetime
        Start date.
    enddate : datetime.datetime
        End date.
    num_chunks : int
        Number of chunks
    max_chunk_len : int
        Maximum chunk length.

    Returns
    -------
    out : list
        List of tuples containing the start and end date of the chunks.
    """
    num_time_steps = int((enddate - startdate).total_seconds() / 300 + 1)

    if num_chunks > num_time_steps:
        num_chunks = num_time_steps

    idx = np.array_split(np.arange(num_time_steps), num_chunks)

    if max_chunk_len is not None or min_chunk_len is not None:
        array_lengths = [len(i) for i in idx]

        if min_chunk_len is not None and np.any(
            np.array(array_lengths) < min_chunk_len
        ):
            idx = np.arange(0, num_time_steps, min_chunk_len)
            if idx[-1] < num_time_steps - 1:
                idx = np.append(idx, [num_time_steps - 1])
            idx = zip(idx[:-1], np.array(idx[1:]) - 1)

        if max_chunk_len is not None and np.any(
            np.array(array_lengths) > max_chunk_len
        ):
            idx = np.arange(0, num_time_steps, max_chunk_len)
            if idx[-1] < num_time_steps - 1:
                idx = np.append(idx, [num_time_steps - 1])
            idx = zip(idx[:-1], np.array(idx[1:]) - 1)

    date_ranges = [
        (
            startdate + timedelta(minutes=int(i[0]) * 5),
            startdate + timedelta(minutes=int((i[-1]) + 1) * 5),
        )
        for i in idx
    ]
    if date_ranges[-1][1] > enddate:
        date_ranges[-1] = (date_ranges[-1][0], enddate)

    return date_ranges
