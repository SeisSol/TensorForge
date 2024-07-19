void trashTheCache(double* trash, int size) {
  for (int i = 0; i < size; ++i) {
    trash[i] += trash[i];
  }
}
