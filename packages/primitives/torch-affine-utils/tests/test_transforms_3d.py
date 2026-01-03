import torch

from torch_affine_utils.transforms_3d import Rx, Ry, Rz, T, S

TRANSFORMS = [Rx, Ry, Rz, T, S]

def test_rotation_around_x():
    """Rotation of y around x should become z."""
    R = Rx(90)
    v = torch.tensor([0, 1, 0, 1]).view((4, 1)).float()
    expected = torch.tensor([0, 0, 1, 1]).view((4, 1)).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)

    R = Rx(90, zyx=True)
    v = torch.tensor([0, 1, 0, 1]).view((4, 1)).float()
    expected = torch.tensor([1, 0, 0, 1]).view((4, 1)).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)


def test_rotation_around_y():
    """Rotation of z around y should be x"""
    R = Ry(90)
    v = torch.tensor([0, 0, 1, 1]).view((4, 1)).float()
    expected = torch.tensor([1, 0, 0, 1]).view((4, 1)).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)

    R = Ry(90, zyx=True)
    v = torch.tensor([1, 0, 0, 1]).view((4, 1)).float()
    expected = torch.tensor([0, 0, 1, 1]).view((4, 1)).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)


def test_rotation_around_z():
    """Rotation of x around z should give y."""
    R = Rz(90)
    v = torch.tensor([1, 0, 0, 1]).view((4, 1)).float()
    expected = torch.tensor([0, 1, 0, 1]).view((4, 1)).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)

    R = Rz(90, zyx=True)
    v = torch.tensor([0, 0, 1, 1]).view((4, 1)).float()
    expected = torch.tensor([0, 1, 0, 1]).view((4, 1)).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)


def test_translation():
    """Translations"""
    M = T([1, 2, 3])
    v = torch.tensor([0, 0, 0, 1]).view((4, 1)).float()
    expected = torch.tensor([1, 2, 3, 1]).view((4, 1)).float()
    assert torch.allclose(M @ v, expected, atol=1e-6)


def test_scaling():
    """Translations"""
    M = S([1, 2, 3])
    v = torch.tensor([1, 1, 1, 1]).view((4, 1)).float()
    expected = torch.tensor([1, 2, 3, 1]).view((4, 1)).float()
    assert torch.allclose(M @ v, expected, atol=1e-6)


def test_devices():
    """Test that the matrices are created on the correct device."""
    for O in TRANSFORMS:
        assert O(torch.tensor(0)).device.type == 'cpu'
        assert O(torch.tensor(0, device="meta")).device.type == 'meta'
        # Whish we could test this, but it doesn't work
        # because we can't grab data from a meta tensor
        #assert O(torch.tensor(0, device="meta"), device="cpu").device.type == 'cpu'
        assert O(torch.tensor(0), device="meta").device.type == 'meta'


def test_batching():
    """Test that the matrices are created with the correct batch size."""

    # Single values should give 4x4 matrices for all transforms
    for O in TRANSFORMS:
        # Test with a single value
        matrix = O(0)
        assert matrix.shape == (4, 4)

    # 1D tensor should give (n,4,4) matrices for all transforms
    for O in [Rx, Ry, Rz]:
        assert O(torch.tensor([0, 90, 180, 250])).shape == (4, 4, 4)

    # 1D tensor with length 3 should result in 4x4 matrices for T and S
    for O in [T, S]:
        matrix = O(torch.tensor([0, 90, 180]))
        assert matrix.shape == (4, 4)
    
    # 1D tensors with length greater than 3 should raise an error for T and S
    for O in [T, S]:
        try:
            O(torch.tensor([0, 90, 180, 250]))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{O.__name__} should raise an error for 1D tensors with length > 3")

    # 2D tensor should give (m,n,4,4) matrices for all rotations
    for O in [Rx, Ry, Rz]:
        assert O(torch.tensor([[0, 90], [180, 250]])).shape == (2, 2, 4, 4)

    # (n,3) tensor should result in (n,4,4) matrices for T and S
    for O in [T, S]:
        matrix = O(torch.tensor([[0, 0, 0], [5, 5, 5]]))
        assert matrix.shape == (2, 4, 4)

    # (n,>3) tensor should raise an error for T and S
    for O in [T, S]:
        try:
            O(torch.tensor([[0, 0, 0, 0], [5, 5, 5, 5]]))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{O.__name__} should raise an error for tensors with last dimension > 3")


def test_backpropagation():
    """Test that gradients can be back propagated from output to input."""
    for O in TRANSFORMS:
        x = torch.tensor([90.0, 60.0, 30.0], requires_grad=True)
        y = O(x)
        assert y.requires_grad

        # y needs to be a scalar value (i.e. a loss) for backpropagation to work
        # hence the sum() operation
        y.sum().backward()
        assert x.grad is not None