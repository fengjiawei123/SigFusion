import torch

def reassemble_patches(patches, original_height, original_width, patch_size):
    """
    Reassemble patch representations into a single image of the original size.

    Args:
    patches (torch.Tensor): The output patches from the transformer model,
                            shape (batch_size, num_patches, patch_height * patch_width * channels)
    original_height (int): The height of the original image.
    original_width (int): The width of the original image.
    patch_size (int): The height and width of each patch (assuming square patches).

    Returns:
    torch.Tensor: Reassembled image of shape (batch_size, channels, original_height, original_width)
    """
    batch_size, num_patches, _ = patches.shape
    num_channels = 1  # Assuming RGB images

    # Assuming the patches are square and the image was evenly divided into patches
    assert original_height % patch_size == 0 and original_width % patch_size == 0
    patches_per_row = original_width // patch_size
    patches_per_col = original_height // patch_size

    # Reshape the patches to make them 2D again
    patches = patches.view(batch_size, num_patches, num_channels, patch_size, patch_size)

    # Initialize the reassembled image
    reassembled_image = torch.zeros((batch_size, num_channels, original_height, original_width), dtype=patches.dtype).to(patches.device)

    # Fill the reassembled image with patches
    patch_idx = 0
    for i in range(patches_per_col):
        for j in range(patches_per_row):
            start_i = i * patch_size
            start_j = j * patch_size
            reassembled_image[:, :, start_i:start_i+patch_size, start_j:start_j+patch_size] = patches[:, patch_idx]
            patch_idx += 1

    return reassembled_image

# # Example usage
# batch_size = 1
# num_patches = 64  # For example, 8x8 patches
# patch_size = 16
# original_height = 128
# original_width = 128
# num_channels = 3

# # Simulate the output from the transformer model (random data for demonstration)
# patches = torch.rand((batch_size, num_patches, num_channels * patch_size * patch_size))

# # Reassemble patches into the original image size
# reassembled_image = reassemble_patches(patches, original_height, original_width, patch_size)
# print("Reassembled Image Shape:", reassembled_image.shape)
