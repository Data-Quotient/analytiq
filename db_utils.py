import subprocess
import sys

from models import Dataset, get_db, DatasetVersion


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install SQLAlchemy if not installed
try:
    from sqlalchemy.orm import Session
except ImportError:
    install_package("SQLAlchemy")
    from sqlalchemy.orm import Session


# Function to add a new dataset to the database
def add_dataset(name, description, filepath):
    db: Session = next(get_db())
    
    try:
        # Create the new dataset entry
        new_dataset = Dataset(name=name, description=description, filepath=filepath)
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)  # Refresh to ensure the object is fully loaded
        
        # Create the default version entry for the new dataset
        default_version = DatasetVersion(
            dataset_id=new_dataset.id,
            version_number="Default Version",
            description="Initial version of the dataset."
        )
        db.add(default_version)
        db.commit()
        
        # Access the attributes you need here while the session is still active
        dataset_info = {
            "name": new_dataset.name,
            "description": new_dataset.description,
            "filepath": new_dataset.filepath
        }
        
        return dataset_info  # Return the relevant dataset information
    except Exception as e:
        db.rollback()
        raise e
