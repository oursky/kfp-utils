import glob
import hashlib
import os
import os.path
import re
from typing import Callable, Dict, List, Optional, Tuple

from smart_open import open

from .executor import BoundedThreadPoolExecutor


def dir_name_mapper(remote_dir: str) -> Callable[[str], str]:
    def mapper(remote_path: str) -> str:
        local_path = remote_path[len(remote_dir) :]  # noqa E203
        if local_path[0] == '/':
            local_path = local_path[1:]
        return local_path

    return mapper


def copy_files_to(
    files_paths: List[str],
    dest_dir: str,
    verbose: bool = True,
    local_name_mapper: Callable[[str], str] = os.path.basename,
    cache_dir: Optional[str] = None,
) -> Dict[str, str]:

    src_dest_map = {
        src: os.path.join(dest_dir, local_name_mapper(src))
        for src in files_paths
    }

    return copy_files(src_dest_map, verbose=verbose, cache_dir=cache_dir)


def copy_files(
    src_dest_map: Dict[str, str], verbose=True, cache_dir: Optional[str] = None
) -> Dict[str, str]:
    if verbose:
        print("Start copying files...")
    futures = []
    with BoundedThreadPoolExecutor() as executor:
        for src, dest in src_dest_map.items():

            def cb(src):
                def wrapped(future):
                    if future.exception() is not None:
                        print(f"Fail to copy {src}.")

                    if verbose:
                        print(f"{src} copied.")

                return wrapped

            future = executor.submit(copy_file, src, dest, cache_dir)
            future.add_done_callback(cb(src))
            futures.append(future)

    return dict(f.result() for f in futures)


def copy_file(
    source_file_path: str,
    destination_file_path: str,
    cache_dir: Optional[str] = None,
) -> Tuple[str, str]:

    if source_file_path.startswith('gs://') and cache_dir:
        source_file_path = cache_file(source_file_path, cache_dir=cache_dir)

    with open(source_file_path, "rb") as source_file:
        if not destination_file_path.startswith("gs://"):
            os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)

        with open(destination_file_path, "wb") as destination_file:
            size = 256 * 1024 * 1024
            while True:
                buf = source_file.read(size)

                if len(buf) == 0:
                    break

                destination_file.write(buf)

    return source_file_path, destination_file_path


def upload_local_dir(local_dir, remote_dir):
    def cb(dest):
        def wrapped(future):
            if future.exception() is not None:
                print(f"Fail to copy to {dest}.")

        return wrapped

    with BoundedThreadPoolExecutor() as executor:
        for file in glob.glob(os.path.join(local_dir, "**"), recursive=True):

            if not os.path.isfile(file):
                continue

            destination_path = remote_dir + file.replace(local_dir, "")
            destination_parent = os.path.dirname(destination_path)

            if (
                not os.path.isdir(destination_parent)
                and re.match(r'^[a-zA-z0-9]+:\/\/', destination_parent) is None
            ):
                try:
                    os.makedirs(destination_parent)
                except FileNotFoundError:
                    print(
                        f"Seems {destination_parent} is not a local path,"
                        "skipped making directory"
                    )
            future = executor.submit(copy_file, file, destination_path)
            future.add_done_callback(cb(destination_path))


def get_cache_path(url, cache_dir='/tmp/cache'):
    m = hashlib.sha256()
    m.update(url.encode())
    return os.path.join(cache_dir, m.hexdigest())


def cache_file(url, cache_dir="/tmp/cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = get_cache_path(url, cache_dir)

    if os.path.exists(cache_path):
        return cache_path

    with open(url, "rb") as in_fp:
        with open(cache_path, "wb") as out_fp:
            out_fp.write(in_fp.read())

    return cache_path
