/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <tuple>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::cout; 
using std::endl; 

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Number of particles 
  num_particles = 1;  

  // Define Gaussian distribution as particle generator
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  // Create particles and initialize them with GPS information
  for (int i = 0; i < num_particles; i++){
    // Create and init particle
    Particle p = Particle(); 
    p.id = i; 
    p.x = dist_x(gen); 
    p.y = dist_y(gen); 
    p.theta = dist_theta(gen); 
    p.weight = 1.0;
    // Add particle and weight to their respective sets
    particles.push_back(p); 
    weights.push_back(p.weight); 
    // DEBUGGING
    /*cout << "Create particle " << p.id << " with (" << p.x << " " << p.y 
         << " " << p.theta << ") and weight " << p.weight << endl; */
  }
  // Set initialization flag
  is_initialized = true;  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
   // For each particle, calculate Particle Movement (Based on velocity and yaw rate measurements)
   for (auto& p : particles){
     double x_f = p.x + (velocity/yaw_rate)*(sin(p.theta + yaw_rate*delta_t) - sin(p.theta)); 
     double y_f = p.y + (velocity/yaw_rate)*(cos(p.theta) - cos(p.theta + yaw_rate*delta_t)); 
     double theta_f = p.theta + yaw_rate*delta_t;
     // Add Gaussian noise and update particle
     std::default_random_engine gen;
     std::normal_distribution<double> dist_x(x_f, std_pos[0]);
     std::normal_distribution<double> dist_y(y_f, std_pos[1]);
     std::normal_distribution<double> dist_theta(theta_f, std_pos[2]);
     p.x = dist_x(gen); 
     p.y = dist_y(gen); 
     p.theta = dist_theta(gen);
     // DEBUGGING
     /*cout << "Move particle " << p.id << " to (" << p.x << " " << p.y 
         << " " << p.theta << ")" << endl; */ 
   }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  // Check each expected LM for the corresponding observation
  // If there's more observations than LMs, observations are gonna be unassigned
  // Later for the probability calculation, this means that the prob of this 
  // measurement got to be 0
  for (auto pred_LM : predicted){
    // Calculate error (i.e. distance) against each observation
    vector < std::tuple<double, LandmarkObs*> > dist_list; 
    for (auto& obs : observations){
      // Check if observation has not been assigned to a LM already
      if (obs.id == 0) {
        double eucl_dist = dist(pred_LM.x, pred_LM.y, obs.x, obs.y);
        dist_list.push_back(std::make_tuple(eucl_dist, &obs)); 
        // DEBUGGING
        //cout << "Distance between LM " << pred_LM.id << " (" << pred_LM.x << " " << pred_LM.y 
        //    << ") and obsevation (" << obs.x << " " << obs.y << ") is " << eucl_dist << endl;    
      }     
    }
    // Sort the distance/error list in ascending order
    // Lowest error element on the top of the vector
    // Assign current LM id to lowest error observation
    if (dist_list.size() != 0){
      sort(dist_list.begin(), dist_list.end()); 
      std::get<1>(dist_list[0])->id = pred_LM.id;
      // DEBUGGING #############################################################
      /*cout << "Sorted List entries:" << endl;  
      for (auto e : dist_list){
        cout << "Observation " << std::get<1>(e)->x << " " << std::get<1>(e)->y 
             << " --> error to LM: " << std::get<0>(e) << endl;  
      }
      cout << "Closest observation to LM " << pred_LM.id << " is (" 
           << std::get<1>(dist_list[0])->x << " " << std::get<1>(dist_list[0])->y << ")" << endl; */   
      // DEBUGGING #############################################################
    }
  }

  //Debugging
  cout << "Observations after Assignment" << endl; 
  for (auto obs : observations){
    cout << "Obervation (" << obs.x << ", " << obs.y <<") is assigned to LM " << obs.id << endl; 
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  // Calculate observation probabilities for each particle
  for (auto& p : particles){
    // (1) TRANSFORM OBSERVATION FROM PARTICLE COS INTO WORLD COS
    vector<LandmarkObs> obs_trans; 
    for (auto obs : observations){
      // Apply transormation matrix
      double x_m = p.x + cos(p.theta)*obs.x - sin(p.theta)*obs.y; 
      double y_m = p.y + sin(p.theta)*obs.x + cos(p.theta)*obs.y; 
      // Store transformed coordinates in obs_trans
      LandmarkObs LM_obs_global = LandmarkObs(); 
      LM_obs_global.x = x_m; 
      LM_obs_global.y = y_m; 
      LM_obs_global.id = 0; 
      obs_trans.push_back(LM_obs_global); 
      // DEBUGGING
      /*cout << "Particle " << p.id << " observes (" << obs.x << " " << obs.y 
           << ") which corresponds to map based coordinates (" 
           << x_m << " " << y_m << ")" << endl; */ 
    }

    // (2) ESTIMATE WHICH LANDMARKS THE PARTICLE IS SUPPOSED TO OBSERVE
    // Define particle sensor range
    double x_min = p.x - sensor_range; 
    double x_max = p.x + sensor_range; 
    double y_min = p.y - sensor_range; 
    double y_max = p.y + sensor_range; 
    // Cut the LMs of interest from the map (Check whether they are within particle range)
    // and push them to the LMs_in_range vector
    vector<LandmarkObs> LMs_in_range; 
    for (auto LM : map_landmarks.landmark_list){
      if (LM.x_f > x_min && LM.x_f < x_max && LM.y_f > y_min && LM.y_f < y_max){
        LandmarkObs LM_in_range = LandmarkObs();
        LM_in_range.x = LM.x_f; 
        LM_in_range.y = LM.y_f; 
        LM_in_range.id = LM.id_i;  
        LMs_in_range.push_back(LM_in_range); 
        //DEBUGGING
        /*cout << "Landmark " << LM.id_i << " (" << LM.x_f << ", " << LM.y_f << ") is in range of particle " 
             << p.id << " (" << p.x << ", " << p.y << " )" << endl; */ 
      }
    }
    // DEBUGGING
    /*cout << "Particle " << p.id << " observes " << observations.size() << " measurements and has "
         << LMs_in_range.size() << "LMs within its range" << endl; */
    
    // (3) ASSOCIATE OBSERVATIONS WITH LANDMARKS 
    dataAssociation(LMs_in_range, obs_trans); 

    // (4) CALCULATE PROBABILITIES 
  }
  


}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}