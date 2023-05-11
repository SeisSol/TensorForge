from gemmforge.exceptions import GenerationError
from gemmforge.matrix.matrix import Matrix


class DenseMatrix(Matrix):
    def __init__(self, num_rows, num_cols, addressing, bbox=None):
        Matrix.__init__(self, num_rows, num_cols, addressing)

        if bbox is not None:
            self.bbox = bbox

            # check whether bbox were given correctly
            coords = [coord for coord in self.bbox]
            if (self.num_rows < self.get_actual_num_rows()) or (
                    self.num_cols < self.get_actual_num_cols()):
                raise GenerationError('Matrix size {}x{} is '
                                      'smaller than bbox {}'.format(self.num_rows,
                                                                    self.num_cols,
                                                                    coords))
            if (self.num_rows < self.bbox[2]) or (self.num_cols < self.bbox[3]):
                raise GenerationError('Bbox {} is '
                                      'outside of Matrix {}x{}'.format(coords,
                                                                       self.num_rows,
                                                                       self.num_cols))
        else:
            self.bbox = (0, 0, num_rows, num_cols)

    def get_actual_num_rows(self):
        return self.bbox[2] - self.bbox[0]

    def get_actual_num_cols(self):
        return self.bbox[3] - self.bbox[1]

    def get_actual_volume(self):
        return self.get_actual_num_rows() * self.get_actual_num_cols()

    def get_real_volume(self):
        return self.num_rows * self.num_cols

    def get_offset_to_first_element(self):
        return self.num_rows * self.bbox[1] + self.bbox[0]

    def __str__(self):
        string = super().__str__()
        string += "bounding box = {}\n".format(self.bbox)
        string += "num. actual rows = {}\n".format(self.get_actual_num_rows())
        string += "num. actual cols = {}\n".format(self.get_actual_num_cols())
        return string