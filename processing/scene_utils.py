from satpy import Scene


def make_scene(msg_file, cth_file, config):
    """Return a Satpy Scene, with or without CTH for parallax correction."""
    if config['parallax']:
        return Scene({
            config['mtg_reader']: [str(msg_file)],
            config['cth_reader']: [str(cth_file)]
        })
    else:
        return Scene(reader=config['mtg_reader'], filenames=msg_file)


def load_and_crop(scene: Scene, channels, roi):
    scene.load(channels)
    crop = scene.crop(ll_bbox=(roi['lon_min'], roi['lat_min'], roi['lon_max'], roi['lat_max']))
    return crop