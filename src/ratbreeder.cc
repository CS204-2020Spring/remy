#include <iostream>

#include "ratbreeder.hh"
#include <cstdlib>
#include <algorithm>

using namespace std;
extern double action_reward_E_array[];
extern int action_count_array[];
const int action_array_size = 13;
const double action_epsilon = 0.1;
extern double last_action_score;
Evaluator< WhiskerTree >::Outcome RatBreeder::improve( WhiskerTree & whiskers )
{
  /* back up the original whiskertree */
  /* this is to ensure we don't regress */
  WhiskerTree input_whiskertree( whiskers );

  /* evaluate the whiskers we have */
  whiskers.reset_generation();
  unsigned int generation = 0;
  unsigned int generation_chosen = 5;
  for(int i = 3; i < action_array_size; i++){
    if(action_reward_E_array[i] > action_reward_E_array[generation_chosen]){
      generation_chosen = i;
    }
  }
  int random_num = rand()%10;
  if((double)random_num <= action_epsilon * 10){
    //random chosen generation
    generation_chosen = (rand() %(action_array_size-3))+3;
  }
  fprintf(stderr," generation chosen: %f\n", (double)generation_chosen);
  


  while ( generation < generation_chosen ) {
    const Evaluator< WhiskerTree > eval( _options.config_range );

    auto outcome( eval.score( whiskers ) );

    /* is there a whisker at this generation that we can improve? */
    auto most_used_whisker_ptr = outcome.used_actions.most_used( generation );

    /* if not, increase generation and promote all whiskers */
    if ( !most_used_whisker_ptr ) {
      generation++;
      whiskers.promote( generation );

      continue;
    }

    WhiskerImprover improver( eval, whiskers, _whisker_options, outcome.score );

    Whisker whisker_to_improve = *most_used_whisker_ptr;

    double score_to_beat = outcome.score;

    while ( 1 ) {
      double new_score = improver.improve( whisker_to_improve );
      assert( new_score >= score_to_beat );
      if ( new_score == score_to_beat ) {
	cerr << "Ending search." << endl;
	break;
      } else {
	cerr << "Score jumps from " << score_to_beat << " to " << new_score << endl;
	score_to_beat = new_score;
      }
    }

    whisker_to_improve.demote( generation + 1 );
    //fprintf(stderr," new score %f, old score %f \n", score_to_beat, outcome.score);

    const auto result __attribute((unused)) = whiskers.replace( whisker_to_improve );
    assert( result );
  }

  /* Split most used whisker */
  apply_best_split( whiskers, generation );

  /* carefully evaluate what we have vs. the previous best */
  const Evaluator< WhiskerTree > eval2( _options.config_range );
  const auto new_score = eval2.score( whiskers, false, 10 );
  const auto old_score = eval2.score( input_whiskertree, false, 10 );


  // the score should be very very small
  // reward in this step is (new_score - old_score.score)
  //double reward_this_step = new_score.score - old_score.score;
  double reward_this_step = max(old_score.score, new_score.score) - last_action_score;
  fprintf(stderr," last_action_score: %f, this time : %f \n", last_action_score, max(old_score.score, new_score.score) );
  action_reward_E_array[generation_chosen] = ((action_reward_E_array[generation_chosen]*action_count_array[generation_chosen]) + reward_this_step ) / (action_count_array[generation_chosen]+1);
  action_count_array[generation_chosen] ++;
  //fprintf(stderr,"new score %f, old score %f \n",new_score.score,  old_score.score);
  last_action_score = max(old_score.score, new_score.score);

  if ( old_score.score >= new_score.score ) {
    fprintf( stderr, "Regression, old=%f, new=%f\n", old_score.score, new_score.score );
    whiskers = input_whiskertree;
    return old_score;
  }

  return new_score;
}

vector< Whisker > WhiskerImprover::get_replacements( Whisker & whisker_to_improve ) 
{
  return whisker_to_improve.next_generation( _options.optimize_window_increment,
                                             _options.optimize_window_multiple,
                                             _options.optimize_intersend );
}
