from services.flat_file_service import FlatFileCreationService
import os


class AssignmentManager:

    def make_flat_file(self):
        # TODO: should use try catch if path does not exist
        if not os.path.isfile('/home/jovyan/work/vectors.csv'):
            flat_file_service = FlatFileCreationService()
            flat_file_service.make_flat_file(
                'file:///home/jovyan/work/GoogleNews-vectors-negative300.bin')

    def manage(self):
        self.make_flat_file()
