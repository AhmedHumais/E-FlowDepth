import numpy
import torch

def plot_points_on_background(points_coordinates,
                              background,
                              points_color=[0, 0, 255]):
    """
    Args:
        points_coordinates: array of (y, x) points coordinates
                            of size (number_of_points x 2).
        background: (3 x height x width)
                    gray or color image uint8.
        color: color of points [red, green, blue] uint8.
    """
    if not (len(background.size()) == 3 and background.size(0) == 3):
        raise ValueError('background should be (color x height x width).')
    _, height, width = background.size()
    background_with_points = background.clone()
    y, x = points_coordinates.transpose(0, 1)
    if len(x) > 0 and len(y) > 0: # There can be empty arrays!
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        if not (x_min >= 0 and y_min >= 0 and x_max < width and y_max < height):
            raise ValueError('points coordinates are outsize of "background" '
                             'boundaries.')
        background_with_points[:, y, x] = torch.Tensor(points_color).type_as(
            background).unsqueeze(-1)
    return background_with_points

def events_to_event_image(event_sequence, height=480, width=640, background=None):
    # polarity = event_sequence[:, 3] == -1.0
    polarity = event_sequence[:, 3] == event_sequence[:, 3].min()

    x_negative = event_sequence[~polarity, 1].astype(numpy.int32)
    y_negative = event_sequence[~polarity, 2].astype(numpy.int32)
    x_positive = event_sequence[polarity, 1].astype(numpy.int32)
    y_positive = event_sequence[polarity, 2].astype(numpy.int32)

    positive_histogram, _, _ = numpy.histogram2d(
        x_positive,
        y_positive,
        bins=(width, height),
        range=[[0, width], [0, height]])
    negative_histogram, _, _ = numpy.histogram2d(
        x_negative,
        y_negative,
        bins=(width, height),
        range=[[0, width], [0, height]])

    # Red -> Negative Events
    red = numpy.transpose((negative_histogram >= positive_histogram) & (negative_histogram != 0))
    # Blue -> Positive Events
    blue = numpy.transpose(positive_histogram > negative_histogram)
    # Normally, we flip first, before we apply the other data augmentations

    if background is None:
        height, width = red.shape
        background = torch.full((3, height, width), 255).byte()

    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(red.astype(numpy.uint8))), background,
        [255, 0, 0])
    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(blue.astype(numpy.uint8))),
        points_on_background, [0, 0, 255])
    return points_on_background
