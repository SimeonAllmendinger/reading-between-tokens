import json
import os
import random
import itertools
import pandas as pd

from typing import Optional, Tuple

from schemas import IssueItem, Statement, IssueOwner
from settings import SETTINGS

class PersonaGenerator():
    def __init__(self, persona_config):
        self.persona_config = persona_config

    def generate_combinations(self) -> list[dict]:
        # Ensure all values in persona_config are lists
        config = {
            key: value if isinstance(value, list) else [value]
            for key, value in self.persona_config.items()
        }

        # Get all keys and corresponding values as lists
        keys = config.keys()
        values = config.values()

        # Generate all combinations
        combinations = [
            dict(zip(keys, combination)) 
            for combination in itertools.product(*values)
        ]

        return combinations


class Country():
    
    def __init__(self, path_data_dir: str, dataset_config: dict):
        self.issue_items = list()
        self.path_data_dir = path_data_dir
        self.dataset_config = dataset_config
        
    def _get_value_probe_variants(
        self,
        statement: str,
        answer: str,
        comment: str
    ) -> list[str]:
        """
        Get the prompt variants for the dataset.
        
        Args:
            statement: str
                The statement.
            answer: str
                The answer.
            comment: str
                The comment.
        
        Returns:
            prompt_variants: list[str]
                The prompt variants.
        """
        value_probe_variants = []
        
        for value_probe_variant in self.dataset_config['value_probe_variants']:
            value_probe_variant = value_probe_variant.replace(
                '$$$statement$$$', statement
            ).replace(
                '$$$answer$$$', answer
            ).replace(
                '$$$comment$$$', comment
            ).replace(
                '$$$io_names$$$', '; '.join(
                random.sample([name for name in self.dataset_config['name_mapping'].keys()],
                len(self.dataset_config['name_mapping'].keys())
                ))
            )
            value_probe_variants.append(value_probe_variant)
        
        return value_probe_variants

    
class Germany(Country):
    
    def __init__(self, path_data_dir: str):
        
        super().__init__(
            path_data_dir=path_data_dir,
            dataset_config=SETTINGS.dataset['GERMANY']
            )
                                                     
        for _election_id in self.dataset_config['Wahl-O-Mat']['election_ids']:

            for _party_name in self.dataset_config['party_names']:
                
                _party = self.get_party(
                        election_id=_election_id, 
                        name=_party_name
                        )
                
                if _party is None:
                    continue
                
                party = IssueOwner(
                    name=_party['longname'],
                    short_name=_party['name'],
                )
                _ = party.get_name_mapped(name_mapping=self.dataset_config['name_mapping'])
                        
                for _statement_id in self.dataset_config['Wahl-O-Mat']['statement_ids']:
        
                    _statement = self.get_statement(election_id=_election_id, _id=_statement_id)
                    
                    statement = Statement(
                        label=_statement['label'],
                        text=_statement['text']
                    )
                
                    comment, answer = self.get_comment_and_answer(
                        election_id=_election_id, 
                        party_id=_party['id'], 
                        statement_id=_statement['id']
                        )

                    prompting_variants = self._get_value_probe_variants(
                        
                        statement=statement.text,
                        answer=answer,
                        comment=comment
                    )

                    for _prompt_idx, _prompt in enumerate(prompting_variants):

                        issue = IssueItem(
                            election_id=_election_id,
                            statement=statement,
                            issue_owner=party,
                            answer=answer,
                            comment=comment,
                            prompt=_prompt,
                            prompt_variant_idx=_prompt_idx,
                        )
                        
                        self.issue_items.append(issue)
                        

    def get_statement(self,
                    election_id: int, 
                    _id: Optional[int]=None,
                    label: Optional[str]=None) -> dict:
        """
        Get the statement for the dataset.
        
        Args:
            election_id: int
                The id of the election.
            label: str
                The label of the statement.
            _id: int
                The id of the statement.
        
        Returns:
            statement: dict
                The statement.
        """
        assert label is not None or _id is not None, "Either label or id must be provided."
        
        statements = json.loads(open(f"{self.path_data_dir}/qual-o-mat-data/data/{election_id}/statement.json").read())
        
        if label is not None:
            statement = [statement for statement in statements if statement['label'] == label][0]
        elif _id is not None:
            statement = [statement for statement in statements if statement['id'] == _id][0]
        
        return statement


    def get_party(self, 
                election_id: int,
                _id: Optional[int]=None,
                name: Optional[str]=None,
                ) -> dict:
        """
        Get the party for the dataset.
        
        Args:
            election_id: int
                The id of the election.
            name: str
                The name of the party.
            _id: int
                The id of the party.
        
        Returns:
            party: dict
                The party.
        """
        assert name is not None or _id is not None, "Either name or id must be provided."
        
        parties = json.loads(open(f"{self.path_data_dir}/qual-o-mat-data/data/{election_id}/party.json").read())
        
        if name is not None:
            party = [party for party in parties if party['name'] in name or name in party['name']]
            
            # if party list is empty, return None
            if not party:
                return None
            
            party = party[0]
            party['name'] = name
            
        elif _id is not None:
            party = [party for party in parties if party['id'] == _id][0]
        
        return party


    def get_comment_and_answer(
        self,
        election_id: int,
        party_id: int,
        statement_id: int
    ) -> tuple[str]:
        """
        Get the comment and answer for the dataset.
        
        Args:
            election_id: int
                The id of the election.
            party_id: int
                The id of the party.
            statement_id: int
                The id of the statement.
        
        Returns:
            comment: str
                The comment.
            answer: str
                The answer.
        """
        
        opinions = json.loads(open(f"{self.path_data_dir}/qual-o-mat-data/data/{election_id}/opinion.json").read())
        comments = json.loads(open(f"{self.path_data_dir}/qual-o-mat-data/data/{election_id}/comment.json").read())
        answers = json.loads(open(f"{self.path_data_dir}/qual-o-mat-data/data/{election_id}/answer.json").read())
        
        # Get the comment and answer
        comment_id, answer_id = [(opinion['comment'], opinion['answer']) for opinion in opinions if opinion['party'] == party_id and opinion['statement'] == statement_id][0]
        comment = str(comments[comment_id]['text'].replace('"',''))
        answer = answers[answer_id]['message']
        
        return comment, answer


class Netherlands(Country):
    
    def __init__(self, path_data_dir: str):

        super().__init__(
            path_data_dir=path_data_dir,
            dataset_config=SETTINGS.dataset['NETHERLANDS']
            )
        
        for _election_id in self.dataset_config['StemWijzer']['election_ids']:
            
            stem_wijzer_df = pd.read_csv(f"{self.path_data_dir}/StemWijzer/{_election_id}.csv", sep=';')
            
            for _statement_id in self.dataset_config['StemWijzer'][_election_id]['statement_ids']:
            
                statement_label = stem_wijzer_df.loc[stem_wijzer_df['no longlist'] == _statement_id, 'thesis short'].values[0]
                statement_text = stem_wijzer_df.loc[stem_wijzer_df['no longlist'] == _statement_id, 'thesis long'].values[0]
                
                statement = Statement(
                    label=statement_label,
                    text=statement_text,
                )
                
                for _party_name in self.dataset_config['StemWijzer'][_election_id]['party_names']:
                    
                    # get key for party name short in config Name_mapping
                    party_long_name = [key for key, value in self.dataset_config['name_mapping'].items() if _party_name in value][0]
                    
                    party = IssueOwner(
                        name=party_long_name,
                        short_name=_party_name,
                    )
                    _ = party.get_name_mapped(name_mapping=self.dataset_config['name_mapping'])
                    
                    comment = stem_wijzer_df.loc[(stem_wijzer_df['no longlist'] == _statement_id) & (stem_wijzer_df['name short'] == _party_name), 'explanation'].values[0]
                    answer = stem_wijzer_df.loc[(stem_wijzer_df['no longlist'] == _statement_id) & (stem_wijzer_df['name short'] == _party_name), 'position text'].values[0]
                    
                    prompting_variants = self._get_value_probe_variants(
                        statement=statement.text,
                        answer=answer,
                        comment=comment
                    )

                    for _prompt_idx, _prompt in enumerate(prompting_variants):
                        
                        issue = IssueItem(
                            election_id=_election_id,
                            statement=statement,
                            issue_owner=party,
                            answer=answer,
                            comment=comment,
                            prompt=_prompt,
                            prompt_variant_idx=_prompt_idx,
                        )
                        
                        self.issue_items.append(issue)


class UnitedStates(Country):
    
    def __init__(self, path_data_dir: str):
        
        super().__init__(
            path_data_dir=path_data_dir,
            dataset_config=SETTINGS.dataset['UNITED STATES']
            )
        
        for _election_id in self.dataset_config['VoteCompass']['election_ids']:
            
            vote_compass_df = pd.read_csv(f"{self.path_data_dir}/VoteCompass/{_election_id}.csv", sep=';')
            
            for _statement_id in self.dataset_config['VoteCompass']['statement_ids']:
            
                statement_label = vote_compass_df.loc[vote_compass_df['ID'] == _statement_id, 'TOPIC'].values[0] + '-' + str(_statement_id)
                statement_text = vote_compass_df.loc[vote_compass_df['ID'] == _statement_id, 'THESIS'].values[0]
                
                statement = Statement(
                    label=statement_label,
                    text=statement_text,
                )
                
                for _io_name in self.dataset_config['VoteCompass'][_election_id]['candidates']:
                    
                    # get key for party name short in config Name_mapping
                    party_long_name = [key for key, value in self.dataset_config['name_mapping'].items() if _io_name in value][0]
                    
                    party = IssueOwner(
                        name=party_long_name,
                        short_name=_io_name,
                    )
                    _ = party.get_name_mapped(name_mapping=self.dataset_config['name_mapping'])
                    
                    comment = vote_compass_df.loc[(vote_compass_df['ID'] == _statement_id) & (vote_compass_df['ISSUE_OWNER'] == _io_name), 'EXPLANATION'].values[0]
                    answer = vote_compass_df.loc[(vote_compass_df['ID'] == _statement_id) & (vote_compass_df['ISSUE_OWNER'] == _io_name), 'RESPONSE'].values[0]
                    
                    prompting_variants = self._get_value_probe_variants(
                        statement=statement.text,
                        answer=answer,
                        comment=comment
                    )

                    for _prompt_idx, _prompt in enumerate(prompting_variants):
                        
                        issue = IssueItem(
                            election_id=_election_id,
                            statement=statement,
                            issue_owner=party,
                            answer=answer,
                            comment=comment,
                            prompt=_prompt,
                            prompt_variant_idx=_prompt_idx,
                        )
                        
                        self.issue_items.append(issue)


class UnitedKingdom(Country):
    
    def __init__(self, path_data_dir: str):
        
        super().__init__(
            path_data_dir=path_data_dir,
            dataset_config=SETTINGS.dataset['UNITED KINGDOM']
            )
        
        for _election_id in self.dataset_config['VoteCompass']['election_ids']:
            
            vote_compass_df = pd.read_csv(f"{self.path_data_dir}/VoteCompass/{_election_id}.csv", sep=';')
            
            for _statement_id in self.dataset_config['VoteCompass']['statement_ids']:
            
                statement_label = vote_compass_df.loc[vote_compass_df['ID'] == _statement_id, 'TOPIC'].values[0] + '-' + str(_statement_id)
                statement_text = vote_compass_df.loc[vote_compass_df['ID'] == _statement_id, 'THESIS'].values[0]
                
                statement = Statement(
                    label=statement_label,
                    text=statement_text,
                )
                
                for _io_name in self.dataset_config['party_names']:
                    
                    # get key for party name short in config Name_mapping
                    party_long_name = [key for key, value in self.dataset_config['name_mapping'].items() if _io_name in value][0]
                    
                    party = IssueOwner(
                        name=party_long_name,
                        short_name=_io_name,
                    )
                    _ = party.get_name_mapped(name_mapping=self.dataset_config['name_mapping'])
                    
                    comment = vote_compass_df.loc[(vote_compass_df['ID'] == _statement_id) & (vote_compass_df['ISSUE_OWNER'] == _io_name), 'EXPLANATION'].values[0]
                    answer = vote_compass_df.loc[(vote_compass_df['ID'] == _statement_id) & (vote_compass_df['ISSUE_OWNER'] == _io_name), 'RESPONSE'].values[0]
                    
                    prompting_variants = self._get_value_probe_variants(
                        statement=statement.text,
                        answer=answer,
                        comment=comment
                    )

                    for _prompt_idx, _prompt in enumerate(prompting_variants):
                        
                        issue = IssueItem(
                            election_id=_election_id,
                            statement=statement,
                            issue_owner=party,
                            answer=answer,
                            comment=comment,
                            prompt=_prompt,
                            prompt_variant_idx=_prompt_idx,
                        )
                        
                        self.issue_items.append(issue)


class Canada(Country):
    
    def __init__(self, path_data_dir: str):
        
        super().__init__(
            path_data_dir=path_data_dir,
            dataset_config=SETTINGS.dataset['CANADA']
            )
        
        for _election_id in self.dataset_config['VoteCompass']['election_ids']:
            
            vote_compass_df = pd.read_csv(f"{self.path_data_dir}/VoteCompass/{_election_id}.csv", sep=';')
            
            for _statement_id in self.dataset_config['VoteCompass']['statement_ids']:
            
                statement_label = vote_compass_df.loc[vote_compass_df['ID'] == _statement_id, 'TOPIC'].values[0] + '-' + str(_statement_id)
                statement_text = vote_compass_df.loc[vote_compass_df['ID'] == _statement_id, 'THESIS'].values[0]
                
                statement = Statement(
                    label=statement_label,
                    text=statement_text,
                )
                
                for _io_name in self.dataset_config['party_names']:
                    
                    # get key for party name short in config Name_mapping
                    party_long_name = [key for key, value in self.dataset_config['name_mapping'].items() if _io_name in value][0]
                    
                    party = IssueOwner(
                        name=party_long_name,
                        short_name=_io_name,
                    )
                    _ = party.get_name_mapped(name_mapping=self.dataset_config['name_mapping'])
                    
                    comment = vote_compass_df.loc[(vote_compass_df['ID'] == _statement_id) & (vote_compass_df['ISSUE_OWNER'] == _io_name), 'EXPLANATION'].values[0]
                    answer = vote_compass_df.loc[(vote_compass_df['ID'] == _statement_id) & (vote_compass_df['ISSUE_OWNER'] == _io_name), 'RESPONSE'].values[0]
                    
                    prompting_variants = self._get_value_probe_variants(
                        statement=statement.text,
                        answer=answer,
                        comment=comment
                    )

                    for _prompt_idx, _prompt in enumerate(prompting_variants):
                        
                        issue = IssueItem(
                            election_id=_election_id,
                            statement=statement,
                            issue_owner=party,
                            answer=answer,
                            comment=comment,
                            prompt=_prompt,
                            prompt_variant_idx=_prompt_idx,
                        )
                        
                        self.issue_items.append(issue)


class NewZealand(Country):
    
    def __init__(self, path_data_dir: str):
        
        super().__init__(
            path_data_dir=path_data_dir,
            dataset_config=SETTINGS.dataset['NEW ZEALAND']
            )
        
        for _election_id in self.dataset_config['VoteCompass']['election_ids']:
            
            vote_compass_df = pd.read_csv(f"{self.path_data_dir}/VoteCompass/{_election_id}.csv", sep=';')
            
            for _statement_id in self.dataset_config['VoteCompass']['statement_ids']:
            
                statement_label = vote_compass_df.loc[vote_compass_df['ID'] == _statement_id, 'TOPIC'].values[0] + '-' + str(_statement_id)
                statement_text = vote_compass_df.loc[vote_compass_df['ID'] == _statement_id, 'THESIS'].values[0]
                
                statement = Statement(
                    label=statement_label,
                    text=statement_text,
                )
                
                for _io_name in self.dataset_config['party_names']:
                    
                    # get key for party name short in config Name_mapping
                    party_long_name = [key for key, value in self.dataset_config['name_mapping'].items() if _io_name in value][0]
                    
                    party = IssueOwner(
                        name=party_long_name,
                        short_name=_io_name,
                    )
                    _ = party.get_name_mapped(name_mapping=self.dataset_config['name_mapping'])
                    
                    comment = vote_compass_df.loc[(vote_compass_df['ID'] == _statement_id) & (vote_compass_df['ISSUE_OWNER'] == _io_name), 'EXPLANATION'].values[0]
                    answer = vote_compass_df.loc[(vote_compass_df['ID'] == _statement_id) & (vote_compass_df['ISSUE_OWNER'] == _io_name), 'RESPONSE'].values[0]
                    
                    prompting_variants = self._get_value_probe_variants(
                        statement=statement.text,
                        answer=answer,
                        comment=comment
                    )

                    for _prompt_idx, _prompt in enumerate(prompting_variants):
                        
                        issue = IssueItem(
                            election_id=_election_id,
                            statement=statement,
                            issue_owner=party,
                            answer=answer,
                            comment=comment,
                            prompt=_prompt,
                            prompt_variant_idx=_prompt_idx,
                        )
                        
                        self.issue_items.append(issue)


  
def get_issue_items(path_data_dir: str, country: str) -> list[IssueItem]:
    """
    Get the issues.
    
    Args:
        path_data_dir: str
            The path to the tmp data directory.
        country: str
            The country of the dataset.
    
    Returns:
        issue_items: list[IssueItem]
            The issues.
        issue_config: dict
            The dataset config.
    """
    
    # Get the issues from the dataset depending on the country
    match country:
        case 'GERMANY':
            wahl_o_mat = Germany(path_data_dir=path_data_dir)
            issue_items = wahl_o_mat.issue_items
        case 'NETHERLANDS':
            stem_wijzer = Netherlands(path_data_dir=path_data_dir)
            issue_items = stem_wijzer.issue_items
        case 'UNITED KINGDOM':
            vote_compass = UnitedKingdom(path_data_dir=path_data_dir)
            issue_items = vote_compass.issue_items
        case 'UNITED STATES':
            vote_compass = UnitedStates(path_data_dir=path_data_dir)
            issue_items = vote_compass.issue_items
        case 'CANADA':
            vote_compass = Canada(path_data_dir=path_data_dir)
            issue_items = vote_compass.issue_items
        case 'NEW ZEALAND':
            vote_compass = NewZealand(path_data_dir=path_data_dir)
            issue_items = vote_compass.issue_items
        case  _:
            raise ValueError(f"Country: {country} not supported.")
    
    # Get the dataset config
    issue_config = SETTINGS.dataset[country]
    
    return issue_items, issue_config
