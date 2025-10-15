import zipfile
import os
import ipynbname
from datetime import datetime

def prepare_submission(submission_dir="Abgaben", m_number="82358",
                       last="Hoegen", first="Jan", date=None, additional=None) -> str:
    """
    Create a ZIP archive of the currently running Jupyter notebook and optionally
    additional files or folders.

    Parameters
    ----------
    submission_dir : str
        Directory where the ZIP will be saved.
    m_number : str
        Matriculation or student number.
    last : str
        Last name.
    first : str
        First name.
    date : str or None
        Datetime string to include in filename. Defaults to current timestamp.
    additional : str | list[str] | None
        Optional single file, folder, or list of paths to include in the ZIP.

    Returns
    -------
    str
        Path to the created ZIP file.
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d_%H%M")

    os.makedirs(submission_dir, exist_ok=True)

    # Get current notebook path and info
    notebook_path = str(ipynbname.path())
    notebook_dir = os.path.dirname(notebook_path)
    _, notebook_filename = os.path.split(notebook_path)
    lab_number = notebook_filename[:2]  # assumes filename starts with lab number

    # Construct output ZIP name
    zip_name = f"{m_number}_{last}_{first}_{lab_number}_{date}.zip"
    destination = os.path.join(submission_dir, zip_name)

    def add_to_zip(zipf, path):
        if os.path.isfile(path):
            arcname = os.path.relpath(path, start=notebook_dir)
            zipf.write(path, arcname=arcname)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=notebook_dir)
                    zipf.write(file_path, arcname=arcname)


    # Write the notebook and any additional files/folders
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(notebook_path, arcname=notebook_filename)

        if additional is not None:
            if isinstance(additional, (str, os.PathLike)):
                add_to_zip(zipf, str(additional))
            elif isinstance(additional, (list, tuple)):
                for item in additional:
                    add_to_zip(zipf, str(item))
            else:
                raise TypeError("`additional` must be a path (str) or list of paths.")

    return destination
