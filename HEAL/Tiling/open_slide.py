# ---------------------------------------------#
# This is the SVS image processing code from the example of openslide
# Some of the codes were referred to Nicolas's script.
#
# ---------------------------------------------#
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
from multiprocessing import Process, JoinableQueue
import numpy as np
import os
import sys
import simplejson as json

VIEWER_SLIDE_NAME = 'slide'


class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds, quality):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._slide = None

    def bk_detect(self, _img):
        """
        Writing the new function for the background detection;
        Remove the image patches which the background is more than 50% of the whole patches.
        """
        try:
            _gray = _img.convert('L')
            bw = _gray.point(lambda x: 0 if x < 220 else 1, 'F')
            arr = np.array(np.asarray(bw))
            avg_bkg = np.average(bw)
            if arr.shape[0] + arr.shape[1] - 2*self._tile_size == 4:
                return avg_bkg
            else:
                return 1

        except Exception as e:
            print(e)
            # print(level, address)
            return 1

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            tile = dz.get_tile(level, address)
            bkg_val = self.bk_detect(tile)
            if bkg_val <= 0.5:
                tile.save(outfile, quality=self._quality)
            self._queue.task_done()

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,
                                 limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, format, associated, queue, t_mag, slide):
        self._dz = dz
        self._basename = basename
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._t_mag = int(t_mag)
        self._slide = slide

    def run(self):
        self._write_tiles_new()
        #self._write_dzi()

    def _write_tiles_new(self):
        _magnification = 20
        _tol = 2
        _factors = self._slide.level_downsamples
        _objective = None
        try:
            _objective = float(self._slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        except Exception as e:
            print(e)
            print(self._basename + " - No Objective information found")
            return
        _available = tuple(_objective / x for x in _factors)
        for level in range(self._dz.level_count-1, -1, -1):
            _thisMag = int(_available[0] / pow(2, self._dz.level_count - (level + 1)))
            if self._t_mag != _thisMag:
                continue
            tile_dir = os.path.join("%s_files" % self._basename, str(_thisMag))
            if not os.path.exists(tile_dir):
                os.makedirs(tile_dir)
            cols, rows = self._dz.level_tiles[level]
            for row in range(rows):
                for col in range(cols):
                    tile_name = os.path.join(tile_dir, '%d_%d.%s' % (col, row, self._format))
                    if not os.path.exists(tile_name):
                        self._queue.put((self._associated, level, (col, row), tile_name))
                    self._tile_done()

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (self._associated or 'slide',
                                                    count, total), end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)

    def _write_dzi(self):
        with open('%s.dzi' % self._basename, 'w') as fh:
            fh.write(self.get_dzi())

    def get_dzi(self):
        return self._dz.get_dzi(self._format)


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, basename, format, tile_size, overlap,
                 limit_bounds, quality, workers, with_viewer, tile_level):
        if with_viewer:
            # Check extra dependency before doing a bunch of work
            import jinja2
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._with_viewer = with_viewer
        self._dzi_data = {}
        self._tile_level = tile_level
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap, limit_bounds, quality).start()

    def run(self):
        self._run_image()
        if self._with_viewer:
            for name in self._slide.associated_images:
                self._run_image(name)
            self._write_html()
            self._write_static()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            if self._with_viewer:
                basename = os.path.join(self._basename, VIEWER_SLIDE_NAME)
            else:
                basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                               limit_bounds=self._limit_bounds)
        tiler = DeepZoomImageTiler(dz, basename, self._format, associated,
                                   self._queue, self._tile_level, image)
        tiler.run()
        self._dzi_data[self._url_for(associated)] = tiler.get_dzi()

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _write_html(self):
        import jinja2
        env = jinja2.Environment(loader=jinja2.PackageLoader(__name__),
                                 autoescape=True)
        template = env.get_template('slide-multipane.html')
        associated_urls = dict((n, self._url_for(n)) for n in self._slide.associated_images)
        try:
            mpp_x = self._slide.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = self._slide.properties[openslide.PROPERTY_NAME_MPP_Y]
            mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            mpp = 0
        # Embed the dzi metadata in the HTML to work around Chrome's
        # refusal to allow XmlHttpRequest from file:///, even when
        # the originating page is also a file:///
        data = template.render(slide_url=self._url_for(None), slide_mpp=mpp,
                               associated=associated_urls, properties=self._slide.properties,
                               dzi_data=json.dumps(self._dzi_data))
        with open(os.path.join(self._basename, 'index.html'), 'w') as fh:
            fh.write(data)

    def _write_static(self):
        basesrc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        basedst = os.path.join(self._basename, 'static')
        self._copydir(basesrc, basedst)
        self._copydir(os.path.join(basesrc, 'images'), os.path.join(basedst, 'images'))

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()
