from abc import ABC, abstractmethod
import copy


class PipeBase(ABC):
    """Abstract base class for a Pipe Component within a pipeline"""

    @abstractmethod
    def __init__(self, **kwargs):
        self.metadata = kwargs.pop('metadata', dict()) or dict()
        self.metadata.update({
            'transformer': '{}.__init__'.format(type(self).__name__),
            'params': kwargs,
        })
        self.validate_dataset()

    def validate_dataset(self):
        """Error checking and type validation."""
        pass

    @abstractmethod
    def load_data(self):
        raise NotImplementedError
                
    @abstractmethod
    def save_data(self):
        """Save this Dataset to disk."""
        raise NotImplementedError
        
    @abstractmethod
    def transform(self):
        """Process loaded data"""
        raise NotImplementedError
        
    @abstractmethod
    def run(self):
        """Run load_date -> transform -> save_data"""
        raise NotImplementedError