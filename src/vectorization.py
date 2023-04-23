import pdf2image
import pytesseract
import sklearn.feature_extraction.text import HashingVector

def get_document_vectors(filenames):
    '''
    Creates document term vectors from a list of PDFs

    Parameters:
    -----------------------------------------
    filenames: List[str]
        List of filenames for each PDF to be transformed
    
    Returns:
    -----------------------------------------
    out: np.ndarray
        2D array of shape (n_docs, n_features) containing the vectors for each document
    '''
    doc_texts = []
    hv = HashingVector(n_features=100)
    for doc in filenames:
        pages = pdf2image.convert_from_path(doc)
        doc_text = ''
        for page in pages:
            doc_text += pytesseract.image_to_string(page).replace('\n',' ')

        doc_texts.append(doc_text)

    return hv.fit_transform(doc_texts)
