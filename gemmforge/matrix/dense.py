from ..exceptions import GenerationError

class DenseMatrix:

    ADDRESSIGN = ["none", "strided", "pointer_based"]
    PTR_TYPES = {"none": "*",
                 "strided": "*",
                 "pointer_based": "**"}

    def __init__(self, num_rows, num_cols, addressing, bbox=None, transpose=False, explicit_offset=False):
        self.name = None
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.transpose = transpose
        self.mutable = None

        if bbox is not None:
            self.bbox = bbox

            # check whether bbox were given correctly
            coords = [coord for coord in self.bbox]
            if (self.num_rows < self.get_actual_num_rows()) or (self.num_cols < self.get_actual_num_cols()):
                raise GenerationError('Matrix size {}x{} is '
                                      'smaller than bbox {}'.format(self.num_rows,
                                                                    self.num_cols,
                                                                    coords))
            if (self.num_rows <= self.bbox[2]) or (self.num_cols <= self.bbox[3]):
                raise GenerationError('Bbox {} is ' 
                                      'outside of Matrix {}x{}'.format(coords,
                                                                       self.num_rows,
                                                                       self.num_cols))

        else:
            self.bbox = (0, 0, num_rows - 1, num_cols - 1)


        if addressing in DenseMatrix.ADDRESSIGN:
            self.addressing = addressing
            self.ptr_type = DenseMatrix.PTR_TYPES[self.addressing]
        else:
            raise ValueError('Invalid matrix addressing. '
                             'Valid types: {}'.format(", ".join(DenseMatrix.ADDRESSIGN)))

    def get_actual_num_rows(self):
        return self.bbox[2] - self.bbox[0] + 1

    def get_actual_num_cols(self):
        return self.bbox[3] - self.bbox[1] + 1

    def get_actual_volume(self):
        return self.get_actual_num_rows() * self.get_actual_num_cols()

    def get_real_volume(self):
        return self.num_rows * self.num_cols

    def get_offset_to_first_element(self):
        return self.num_rows * self.bbox[1] + self.bbox[0]

    def _set_name(self, name):
        self.name = name

    def _set_mutability(self, is_mutable):
        self.mutable = is_mutable

    def is_mutable(self):
        if self.mutable is not None:
            return self.mutable
        else:
            raise ValueError("mutability has not been set")

    def __str__(self):
        string = "num. rows = {}\n".format(self.num_rows)
        string += "num. columns = {}\n".format(self.num_cols)
        string += "bounding box = {}\n".format(self.bbox)
        string += "addressing = {}\n".format(self.addressing)
        string += "num. actual rows = {}\n".format(self.get_actual_num_rows())
        string += "num. actual cols = {}\n".format(self.get_actual_num_cols())
        string += "transpose = {}\n".format(self.transpose)
        return string

