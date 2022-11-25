
# locals
from .helper import to_device

# basics
import pandas as pd

# image
from pdf2image import convert_from_path

# ml
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


IMAGE_SIZE = 1280


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device."""

    def __init__(self, dl, device):
      """Initialize and save variables."""
      self.dl = dl
      self.device = device

    def __iter__(self):
      """Yield a batch of data after moving it to device."""
      for b in self.dl:
        yield to_device(b, self.device)

    def __len__(self):
      """Get Number of batches."""
      return len(self.dl)


class ImageDataset(Dataset):
  def __init__(self, file_name_annotations, folder_name_pdfs, transform=True):

    # read annotations
    self.annotations = pd.read_csv(f'data/{file_name_annotations}.csv')[:5]
    self.root_dir = f'data/{folder_name_pdfs}'

    # with h_pdf=210 und w_pdf=297
    transform_to_size = int(210 * IMAGE_SIZE / 297)
    transform_compose = transforms.Compose([transforms.Resize(size=transform_to_size, max_size=IMAGE_SIZE),
                                            SquarePad(),
                                            transforms.ToTensor()])

    # 210*max size / 297 mit h_pdf=210 und w_pdf=297
    self.transform = transform_compose if transform else False

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):

    if self.transform:
      file_name = index  # self.annotations.at[index, 'file_name']
      img_path = self.root_dir + f'/{file_name}.pdf'

      page_image = convert_from_path(img_path)[0]
      image_transformed = self.transform(page_image)

    else:
      img_path = self.root_dir + f'/{index}'
      image_transformed = torch.load(img_path)

    y_label = self.encode_label(self.annotations.at[index, 'sdgs'])

    return (image_transformed, y_label)

  def encode_label(self, sdgs):
    """Encoding the classes into a tensor of shape (17) with 0 and 1s."""
    target = torch.zeros(17)

    if str(sdgs) != 'nan':
      for sdg in str(sdgs).split(','):
        label = int(sdg) - 1
        target[label] = 1

    return target


class PredictionDataset(Dataset):
  def __init__(self, file_name, transform=None):

    # read annotations
    file_path = 'data/sustainability_reports_500_1500/' + str(file_name)

    # with h_pdf=210 und w_pdf=297
    transform_to_size = int(210 * IMAGE_SIZE / 297)
    transform_compose = transforms.Compose([transforms.Resize(size=transform_to_size, max_size=IMAGE_SIZE),
                                            SquarePad(),
                                            transforms.ToTensor()])

    self.transform = transform_compose if transform else False

    # open PDF-Reader
    self.pdf = convert_from_path(file_path)

  def __len__(self):
    return len(self.pdf)

  def __getitem__(self, index):

    # get pdf pages as png
    img = self.pdf[index]

    # transform image
    image_transformed = self.transform(img)

    return (image_transformed, torch.zeros(17))



class SquarePad:
  def __call__(self, image):
    max_w = IMAGE_SIZE
    max_h = IMAGE_SIZE

    imsize = image.size
    h_padding = (max_w - imsize[0]) / 2
    v_padding = (max_h - imsize[1]) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5

    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))

    return F.pad(image, padding, 0, 'constant')