from faker import Faker
from convertbng.util import convert_bng
import pandas as pd


class FakerGB(Faker):
    def __init__(self, locale='en_GB'):
        super().__init__(locale)

    def mk_first_name(self): return self.first_name()

    def mk_last_name(self): return self.last_name()

    def mk_name(self): return self.name()

    def mk_address(self): return self.address()

    def mk_email(self): return self.email()

    def mk_url(self): return self.url()

    def mk_image_url(self): return self.image_url()


def lonlat2bng(df, long='longitude', lat: str = 'latitude') -> pd.DataFrame:
    """covert longlat columns to eastern northern and return them as part of the DataFrame"""
    df['eastern'], df['northern'] = convert_bng(df[long].tolist(), df[lat].tolist())
    return df
