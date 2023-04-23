import vectorization

class PlagiarismChecker:
    def __init__(self, filenames):
        self.doc_matrix = vectorization.get_document_vectors(filenames)
