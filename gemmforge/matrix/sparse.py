from gemmforge.exceptions import GenerationError
from gemmforge.matrix.matrix import Matrix
import json

# Cordinate object form needs be a dictionary of the following entries:
# rows - number of rows (int) e.g. "rows" : 2
# cols - number of columns (int) e.g. "cols": 2
# entries - and array of triplet arrayys of row,col,value, 0-indexed e.g. "entries" : [[0,0,"1.0"],[1,1,"1.0"]]
# Optionally a name can be provided e.g. "name": "simple_identity"
# Final example : {"name": "simple_identity", "rows": 2, "cols": 2, "entries": [[0,0,"1.0"],[1,1,"1.0"]]}

# Right now the values are discarded and the locations of non-zeros are preserved
# Coo_matrix is either the json read into a python dict of the string representation of json


class SparseMatrix(Matrix):
    def __init__(self, num_rows, num_cols, addressing, coo_matrix):
        Matrix.__init__(self, num_rows, num_cols, addressing)
        self.elcount = 0

        if type(coo_matrix) != str and type(coo_matrix) != dict:
            raise GenerationError(
                "Provided coo_matrix to SparseMatrix needs to be a json read as dict or str representation of a json")

        if type(coo_matrix) == str:
            try:
                coo_matrix = json.loads(coo_matrix)
            except:
                raise GenerationError(
                    "Provided coo_matrix to SparseMatrix has type string but it is not a valid string representation of the json")

        rows = int(coo_matrix["rows"])
        cols = int(coo_matrix["cols"])
        coo_triplets = coo_matrix["entries"]

        if "name" in coo_matrix.keys():
            name = coo_matrix["name"]
            self.set_name(name)

        if num_rows > 0 and num_rows != rows:
            GenerationError("Given number of rows does not match the provided json sparse matrix format")
        if num_rows <= 0:
            num_rows = rows
        if num_cols > 0 and num_cols != cols:
            GenerationError("Given number of columns does not match the provided json sparse matrix format")
        if num_cols <= 0:
            num_cols = cols

        self.dense_representation = [[0] * num_cols for _ in range(num_rows)]
        for triplet in coo_triplets:
            self.dense_representation[int(triplet[0])][int(triplet[1])] = float(triplet[2])

        self.coo = coo_triplets

        self.coo_row_major = [[] for _ in range(num_rows)]
        self.coo_col_major = [[] for _ in range(num_cols)]
        for triplet in coo_triplets:
            self.coo_row_major[int(triplet[0])].append(int(triplet[1]))
            self.coo_col_major[int(triplet[1])].append(int(triplet[0]))
            self.elcount += 1
        # If the triplets are not sorted, we need to generated iteration orders, during the generation we need
        # to find the offsets of the elements
        for row in self.coo_row_major:
            row.sort()
        for col in self.coo_col_major:
            col.sort()

        # With this I extract the values into a single list where the coordinates will be saved in the same order in
        # coo_row_major and coo_col_major
        entriescopy = coo_matrix["entries"]
        entriescopy.sort(key=lambda x: x[0] * num_cols + x[1])
        self.values_row_major = [i[2] for i in entriescopy]
        entriescopy = coo_matrix["entries"]
        entriescopy.sort(key=lambda x: x[1] * num_rows + x[0])
        self.values_col_major = [i[2] for i in entriescopy]

    def get_actual_num_rows(self):
        return self.num_rows

    def get_actual_num_cols(self):
        return self.num_cols

    def get_actual_volume(self):
        return self.num_rows * self.num_cols

    def get_real_volume(self):
        return self.get_el_count()

    def get_offset_to_first_element(self):
        return 0

    def get_matrix_type(self):
        return "sparse"

    def __str__(self):
        string = super().__str__()
        string += str(self.dense_representation)
        return string

    def get_coo_row_major(self):
        return self.coo_row_major

    def get_coo_col_major(self):
        return self.coo_col_major

    def get_coo(self):
        return self.coo

    def get_values_row_major(self):
        return self.values_row_major

    def get_values_col_major(self):
        return self.values_col_major

    def get_el_count(self):
        return self.elcount