"""Tests for action_valuation module using real StatsBomb data."""

import pytest
import pandas as pd
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader

from src.action_valuation import calculate_action_values
from src.xthreat import get_default_xt_model


@pytest.fixture(scope="module")
def statsbomb_loader():
    """Initialize StatsBombLoader for real data."""
    return StatsBombLoader(root="data/statsbomb/data", getter="local")


@pytest.fixture(scope="module")
def sample_game(statsbomb_loader):
    """Load a sample game from StatsBomb data."""
    games = statsbomb_loader.games(competition_id=43, season_id=3)
    game = games.iloc[0]
    events = statsbomb_loader.events(game['game_id'])
    return game, events


@pytest.fixture(scope="module")
def xt_model():
    """Load default xT model."""
    return get_default_xt_model()


def find_possession_ending_with_goal(events: pd.DataFrame):
    """Find a possession that ends with a goal."""
    for possession_id, possession_group in events.groupby('possession'):
        possession_team = possession_group['possession_team_name'].iloc[0]
        team_events = possession_group[possession_group['team_name'] == possession_team]

        if len(team_events) >= 3:
            last_event = team_events.iloc[-1]
            if last_event['type_name'] == 'Shot':
                extra = last_event.get('extra', {})
                if isinstance(extra, dict) and 'shot' in extra:
                    outcome = extra['shot'].get('outcome', {})
                    if isinstance(outcome, dict) and outcome.get('name') == 'Goal':
                        return team_events
    return None


def find_possession_ending_with_shot_no_goal(events: pd.DataFrame):
    """Find a possession that ends with a shot (not a goal)."""
    for possession_id, possession_group in events.groupby('possession'):
        possession_team = possession_group['possession_team_name'].iloc[0]
        team_events = possession_group[possession_group['team_name'] == possession_team]

        if len(team_events) >= 3:
            last_event = team_events.iloc[-1]
            if last_event['type_name'] == 'Shot':
                extra = last_event.get('extra', {})
                if isinstance(extra, dict) and 'shot' in extra:
                    outcome = extra['shot'].get('outcome', {})
                    if isinstance(outcome, dict) and outcome.get('name') != 'Goal':
                        return team_events
    return None


class TestCalculateActionValuesGoal:
    """Test calculate_action_values for possessions ending with a goal."""

    def test_goal_value_is_one(self, sample_game, xt_model):
        """Test that a goal action has value 1.0."""
        game, events = sample_game

        # Find possession ending with goal
        possession = find_possession_ending_with_goal(events)

        if possession is None:
            pytest.skip("No possession ending with goal found in sample data")

        # Convert to SPADL actions
        actions = spadl.statsbomb.convert_to_actions(
            possession,
            game['home_team_id'],
            xy_fidelity_version=2
        )

        # Calculate action values
        values = calculate_action_values(actions, possession, xt_model)

        # Add type names for verification
        actions = spadl.add_names(actions)

        # Assert last action (goal) has value 1.0
        assert len(values) == len(actions), "Values length should match actions length"
        assert values.iloc[-1] == 1.0, f"Goal action should have value 1.0, got {values.iloc[-1]}"

        # Verify it's indeed a shot action
        assert actions.iloc[-1]['type_name'] == 'shot', "Last action should be a shot"

        print(f"\n✓ Goal test passed:")
        print(f"  Possession length: {len(actions)} actions")
        print(f"  Last action type: {actions.iloc[-1]['type_name']}")
        print(f"  Last action value: {values.iloc[-1]}")

    def test_goal_possession_all_values_calculated(self, sample_game, xt_model):
        """Test that all actions in goal possession have calculated values."""
        game, events = sample_game

        possession = find_possession_ending_with_goal(events)

        if possession is None:
            pytest.skip("No possession ending with goal found in sample data")

        actions = spadl.statsbomb.convert_to_actions(
            possession,
            game['home_team_id'],
            xy_fidelity_version=2
        )

        values = calculate_action_values(actions, possession, xt_model)

        # Add type names for verification
        actions = spadl.add_names(actions)

        # Check all values are not NaN
        assert not values.isna().any(), "All actions should have non-NaN values"

        # Check values are Series with correct index
        assert isinstance(values, pd.Series), "Values should be a pandas Series"
        assert len(values) == len(actions), "Values should match actions length"

        # Print summary
        print(f"\n✓ All values calculated for goal possession:")
        print(f"  Total actions: {len(actions)}")
        print(f"  Action types: {actions['type_name'].value_counts().to_dict()}")
        print(f"  Value range: [{values.min():.4f}, {values.max():.4f}]")
        print(f"  Last action (goal) value: {values.iloc[-1]}")


class TestCalculateActionValuesShot:
    """Test calculate_action_values for possessions ending with a shot (no goal)."""

    def test_shot_value_is_xg(self, sample_game, xt_model):
        """Test that a shot (no goal) action has value equal to xG."""
        game, events = sample_game

        # Find possession ending with shot (no goal)
        possession = find_possession_ending_with_shot_no_goal(events)

        if possession is None:
            pytest.skip("No possession ending with shot (no goal) found in sample data")

        # Get expected xG value from original event
        last_event = possession.iloc[-1]
        extra = last_event.get('extra', {})
        expected_xg = extra['shot']['statsbomb_xg']

        # Convert to SPADL actions
        actions = spadl.statsbomb.convert_to_actions(
            possession,
            game['home_team_id'],
            xy_fidelity_version=2
        )

        # Calculate action values
        values = calculate_action_values(actions, possession, xt_model)

        # Add type names for verification
        actions = spadl.add_names(actions)

        # Assert last action (shot) has value equal to xG
        assert len(values) == len(actions), "Values length should match actions length"
        assert values.iloc[-1] == pytest.approx(expected_xg, abs=1e-6), \
            f"Shot action should have value {expected_xg}, got {values.iloc[-1]}"

        # Verify it's indeed a shot action
        assert actions.iloc[-1]['type_name'] == 'shot', "Last action should be a shot"

        # Verify it's not a goal
        outcome = extra['shot']['outcome']['name']
        assert outcome != 'Goal', f"This should not be a goal, got outcome: {outcome}"

        print(f"\n✓ Shot (no goal) test passed:")
        print(f"  Possession length: {len(actions)} actions")
        print(f"  Last action type: {actions.iloc[-1]['type_name']}")
        print(f"  Expected xG: {expected_xg:.4f}")
        print(f"  Calculated value: {values.iloc[-1]:.4f}")
        print(f"  Shot outcome: {outcome}")

    def test_shot_value_between_zero_and_one(self, sample_game, xt_model):
        """Test that shot xG value is between 0 and 1."""
        game, events = sample_game

        possession = find_possession_ending_with_shot_no_goal(events)

        if possession is None:
            pytest.skip("No possession ending with shot (no goal) found in sample data")

        actions = spadl.statsbomb.convert_to_actions(
            possession,
            game['home_team_id'],
            xy_fidelity_version=2
        )

        values = calculate_action_values(actions, possession, xt_model)

        # Check shot value is in valid range
        shot_value = values.iloc[-1]
        assert 0.0 <= shot_value <= 1.0, \
            f"Shot xG value should be between 0 and 1, got {shot_value}"

        print(f"\n✓ Shot value range test passed:")
        print(f"  Shot value: {shot_value:.4f}")
        print(f"  Valid range: [0.0, 1.0]")

    def test_multiple_shots_have_different_xg(self, statsbomb_loader, xt_model):
        """Test that different shots have different xG values."""
        # Load game
        games = statsbomb_loader.games(competition_id=43, season_id=3)
        game = games.iloc[0]
        events = statsbomb_loader.events(game['game_id'])

        # Find multiple shot possessions
        shot_possessions = []
        for possession_id, possession_group in events.groupby('possession'):
            possession_team = possession_group['possession_team_name'].iloc[0]
            team_events = possession_group[possession_group['team_name'] == possession_team]

            if len(team_events) >= 3:
                last_event = team_events.iloc[-1]
                if last_event['type_name'] == 'Shot':
                    extra = last_event.get('extra', {})
                    if isinstance(extra, dict) and 'shot' in extra:
                        shot_possessions.append(team_events)
                        if len(shot_possessions) >= 3:
                            break

        if len(shot_possessions) < 2:
            pytest.skip("Not enough shot possessions found")

        shot_values = []
        for possession in shot_possessions:
            actions = spadl.statsbomb.convert_to_actions(
                possession,
                game['home_team_id'],
                xy_fidelity_version=2
            )
            values = calculate_action_values(actions, possession, xt_model)
            shot_values.append(values.iloc[-1])

        # Check that not all shots have the same xG
        unique_values = len(set(shot_values))
        assert unique_values > 1, \
            f"Expected different xG values for different shots, got {unique_values} unique value(s)"

        print(f"\n✓ Multiple shots test passed:")
        print(f"  Number of shots tested: {len(shot_values)}")
        print(f"  Unique xG values: {unique_values}")
        print(f"  xG values: {[f'{v:.4f}' for v in shot_values]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

