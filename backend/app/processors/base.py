class BaseProcessor:
    def __init__(self, pdf_path, debug=False, use_cache=False, export_filename="financial_statement.csv"):
        self.pdf_path = pdf_path
        self.debug = debug
        self.use_cache = use_cache
        self.export_filename = export_filename

    def process(self):
        raise NotImplementedError("Subclasses must implement `process`")
