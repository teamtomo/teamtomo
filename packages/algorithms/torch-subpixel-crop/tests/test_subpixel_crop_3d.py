import torch

from torch_subpixel_crop import subpixel_crop_3d


def test_subpixel_crop_single_3d():
    image = torch.zeros((10, 10, 10))
    image[4:6, 4:6, 4:6] = 1

    cropped_image = subpixel_crop_3d(
        image=image,
        positions=torch.tensor([5, 5, 5]).float(),
        sidelength=4
    )
    assert cropped_image.shape == (4, 4, 4)

    expected = torch.zeros((4, 4, 4))
    expected[1:3, 1:3, 1:3] = 1
    assert torch.allclose(cropped_image, expected)


def test_subpixel_crop_single_3d_return_rfft():
    image = torch.zeros((10, 10, 10))
    image[4:6, 4:6, 4:6] = 1

    cropped_image_real = subpixel_crop_3d(
        image=image,
        positions=torch.tensor([5.5, 5.5, 5.5]).float(),
        sidelength=4
    )

    cropped_image_rfft = subpixel_crop_3d(
        image=image,
        positions=torch.tensor([5.5, 5.5, 5.5]).float(),
        sidelength=4,
        return_rfft=True,
        decenter=False,
    )
    cropped_image_rfft = torch.fft.irfftn(cropped_image_rfft, s=(4, 4, 4))
    assert torch.allclose(cropped_image_rfft, cropped_image_real)


def test_subpixel_crop_single_3d_return_rfft_decenter():
    image = torch.zeros((10, 10, 10))
    image[4:6, 4:6, 4:6] = 1

    cropped_image_real = subpixel_crop_3d(
        image=image,
        positions=torch.tensor([5.5, 5.5, 5.5]).float(),
        sidelength=4
    )

    cropped_image_decenter = subpixel_crop_3d(
        image=image,
        positions=torch.tensor([5.5, 5.5, 5.5]).float(),
        sidelength=4,
        return_rfft=True,
        decenter=True,
    )
    cropped_image_decenter = (
        torch.fft.irfftn(cropped_image_decenter, s=(4, 4, 4))
    )
    cropped_image_decenter = (
        torch.fft.fftshift(cropped_image_decenter, dim=(-3, -2, -1))
    )
    assert torch.allclose(cropped_image_decenter, cropped_image_real)


def test_subpixel_crop_multi_3d():
    image = torch.zeros((10, 10, 10))
    image[4:6, 4:6, 4:6] = 1

    cropped_image = subpixel_crop_3d(
        image=image,
        positions=torch.tensor([[4, 4, 4], [5, 5, 5]]).float(),
        sidelength=4
    )
    assert cropped_image.shape == (2, 4, 4, 4)

    expected_0 = torch.zeros((4, 4, 4))
    expected_0[2:4, 2:4, 2:4] = 1
    assert torch.allclose(cropped_image[0], expected_0)

    expected_1 = torch.zeros((4, 4, 4))
    expected_1[1:3, 1:3, 1:3] = 1
    assert torch.allclose(cropped_image[1], expected_1)
