import torch

from torch_affine_utils.transforms_2d import R, T, S

TRANSFORMS = [R, T, S]

def test_rotation():
    """Rotating x-axis by 90 degrees should become y-axis."""
    # Standard coordinate system (xyw)
    rotation = R(90)
    v = torch.tensor([1, 0, 1]).view((3, 1)).float()
    expected = torch.tensor([0, 1, 1]).view((3, 1)).float()
    assert torch.allclose(rotation @ v, expected, atol=1e-6)

    # Test with yx coordinate system
    rotation = R(90, yx=True)
    v = torch.tensor([0, 1, 1]).view((3, 1)).float()
    expected = torch.tensor([1, 0, 1]).view((3, 1)).float()
    assert torch.allclose(rotation @ v, expected, atol=1e-6)


def test_translation():
    """Translation should add the offset to the position."""
    translation = T([2, 3])
    v = torch.tensor([1, 1, 1]).view((3, 1)).float()
    expected = torch.tensor([3, 4, 1]).view((3, 1)).float()
    assert torch.allclose(translation @ v, expected, atol=1e-6)


def test_scaling():
    """Scaling should multiply each coordinate by the scale factor."""
    scaling = S([2, 3])
    v = torch.tensor([1, 1, 1]).view((3, 1)).float()
    expected = torch.tensor([2, 3, 1]).view((3, 1)).float()
    assert torch.allclose(scaling @ v, expected, atol=1e-6)


def test_devices():
    """Test that the matrices are created on the correct device."""
    for O in TRANSFORMS:
        assert O(torch.tensor(0)).device.type == 'cpu'
        assert O(torch.tensor(0,device="meta")).device.type == 'meta'
        # Whish we could test this, but it doesn't work
        # because we can't grab data from a meta tensor
        #assert O(torch.tensor(0,device="meta"),device="cpu").device.type == 'cpu'
        assert O(torch.tensor(0),device="meta").device.type == 'meta'


def test_batching():
    """Test that the matrices are created with the correct batch size."""

    # Single values should give 3x3 matrices for all transforms
    for O in TRANSFORMS:
        # Test with a single value
        matrix = O(0)
        assert matrix.shape == (3, 3)

    # 1D tensor should give (n,3,3) matrices for all transforms
    assert R(torch.tensor([0, 90, 180, 250])).shape == (4, 3, 3)

    # 1D tensor with length 2 should result in 3x3 matrices for T and S
    for O in [T, S]:
        matrix = O(torch.tensor([0, 90]))
        assert matrix.shape == (3, 3)
    
    # 1D tensors with length greater than 2 should raise an error for T and S
    for O in [T, S]:
        try:
            O(torch.tensor([0, 90, 180]))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{O.__name__} should raise an error for 1D tensors with length > 2")

    # 2D tensor should give (m,n,3,3) matrices for all rotations
    assert R(torch.tensor([[0, 90], [180, 250]])).shape == (2, 2, 3, 3)

    # (n,2) tensor should result in (n,3,3) matrices for T and S
    for O in [T, S]:
        matrix = O(torch.tensor([[0, 0], [5, 5]]))
        assert matrix.shape == (2, 3, 3)

    # (n,>2) tensor should raise an error for T and S
    for O in [T, S]:
        try:
            O(torch.tensor([[0, 0, 0], [5, 5, 5]]))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{O.__name__} should raise an error for tensors with last dimension > 2")


def test_backpropagation_gradients():
    """Test that gradients can be back propagated from output to input."""
    for O in TRANSFORMS:
        x = torch.tensor([90.0, 45.0], requires_grad=True)
        y = O(x)
        assert y.requires_grad

        # y needs to be a scalar value (i.e. a loss) for backpropagation to work
        # hence the sum() operation
        y.sum().backward()
        assert x.grad is not None