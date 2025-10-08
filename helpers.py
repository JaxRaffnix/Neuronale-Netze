# helpers.py
import zipfile
import os
import ipynbname
from datetime import datetime

def prepare_submission(submission_dir="Abgaben", m_number="82358", last="Hoegen", first="Jan", date=None) -> str:
    """
    Create a zip of the currently running Jupyter notebook with a new filename
    of format {submission_dir}/{m_number}_{last}_{first}_{lab_number}_{YYYYMMDD_HHMM}.zip.

    Parameters
    ----------

    Returns
    -------
    str
        Path to the new notebook.
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(submission_dir, exist_ok=True)

    notebook_path = str(ipynbname.path())
    directory, notebook_filename = os.path.split(notebook_path)
    lab_number = notebook_filename[:2]

    filename = f"{m_number}_{last}_{first}_{lab_number}_{date}.zip"
    destination = os.path.join(submission_dir, filename)
    # destination = os.path.join(submission_dir, destination)

    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(notebook_path, arcname=notebook_filename)

    return destination

