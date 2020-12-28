class Error(Exception):
    """Base Exception Class

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Custom Error"):
        self.message = message
        super().__init__(self.message)

class ImageFileNotFoundError(Error):
    def __init__(self, message = "Image Not found in the specified Dataset path"):
        super().__init__(message=message)


class ImageProcessingError(Error):
    def __init__(self, message = "Image Not found in the specified Dataset path"):
        super().__init__(message=message)

class DatasetDirectoryNotFoundError(Error):
    def __init__(self, message = "Invalid dataset path"):
        super().__init__(message=message)

class DataPreprocessingError(Error):
    def __init__(self, message = "Data Preprocessing cannot be done. Code error"):
        super().__init__(message=message)

class InvalidDatasetError(Error):
    def __init__(self, message = "Invalid dataset"):
        super().__init__(message=message)

class PredictionError(Error):
    def __init__(self, message = "Invalid dataset"):
        super().__init__(message=message)