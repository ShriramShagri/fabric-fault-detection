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