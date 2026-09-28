import argparse
import hashlib
from pathlib import Path


SUPPORTED_EXTENSIONS = {
	".jpg",
	".jpeg",
	".png",
	".bmp",
	".gif",
	".tif",
	".tiff",
	".webp",
}


def hash_file(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
	"""Return the SHA-256 hash of a file."""
	hasher = hashlib.sha256()
	with file_path.open("rb") as f:
		while True:
			chunk = f.read(chunk_size)
			if not chunk:
				break
			hasher.update(chunk)
	return hasher.hexdigest()


def is_image_file(path: Path) -> bool:
	return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def find_matching_images(folder: Path, target_image: Path) -> list[Path]:
	target_hash = hash_file(target_image)
	matches: list[Path] = []

	for file_path in folder.rglob("*"):
		if not is_image_file(file_path):
			continue

		# Skip the exact same file path as target if it's inside the folder.
		if file_path.resolve() == target_image.resolve():
			continue

		try:
			print(f"Checking: {file_path}")
			if hash_file(file_path) == target_hash:
				matches.append(file_path)
		except OSError:
			# Ignore unreadable files and continue scanning.
			print(f"Warning: Could not read file: {file_path}")
			continue

	print(f"folder: {folder}, target_image: {target_image}, matches: {matches}")

	return matches


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Find images in a folder that exactly match a given image "
			"by comparing SHA-256 hashes."
		)
	)
	parser.add_argument(
		"inputs",
		nargs="*",
		help=(
			"Either: <folder> <image> or labeled form: "
			"folder <folder> image <image>"
		),
	)
	parser.add_argument("--folder", type=Path, help="Path to folder containing images")
	parser.add_argument("--image", type=Path, help="Path to target image")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	folder: Path | None = args.folder
	image: Path | None = args.image

	if folder is None and image is None:
		raw_inputs = args.inputs

		if len(raw_inputs) == 2:
			folder = Path(raw_inputs[0])
			image = Path(raw_inputs[1])
		elif (
			len(raw_inputs) == 4
			and raw_inputs[0].lower() == "folder"
			and raw_inputs[2].lower() == "image"
		):
			folder = Path(raw_inputs[1])
			image = Path(raw_inputs[3])
		else:
			raise SystemExit(
				"Error: use either '<folder> <image>' or "
				"'folder <folder> image <image>'"
			)
	elif folder is None or image is None:
		raise SystemExit("Error: --folder and --image must be provided together")

	if not folder.exists() or not folder.is_dir():
		raise SystemExit(f"Error: folder not found or not a directory: {folder}")

	if not image.exists() or not image.is_file():
		raise SystemExit(f"Error: image file not found: {image}")

	if not is_image_file(image):
		raise SystemExit(
			f"Error: target image extension not supported ({image.suffix})"
		)

	matches = find_matching_images(folder, image)

	if not matches:
		print("No matching images found.")
		return

	print("Matching images:")
	for match in matches:
		print(f"Name: {match.name}")
		print(f"Path: {match.resolve()}")
		print("-")


if __name__ == "__main__":
	main()
