! cd CompSci/Projects/ReinforcementLearning
! gfortran -o output RL2pt6.f95 -lpdflib -lranlib -lrnglib
! ./output

program reinforce

  implicit none

  character(10) :: timeval1, timeval2, temptime
  integer :: i, j, k ! incrementors
  integer, parameter :: simlength = 200, steplen = 200000 ! 2000, 1000
  integer :: ierror ! necessary for output
  integer, parameter :: epslen = 6, ucblen = 7, gbalen = 8, oivlen = 5, qlen = 10
  real(kind=8), dimension(epslen) :: epsg, avgReps
  real(kind=8), dimension(ucblen) :: cs, avgRucb
  real(kind=8), dimension(gbalen) :: alphas, avgRgba
  real(kind=8), dimension(oivlen) :: qs, avgRoiv
  real(kind=8), dimension(qlen) :: q_stars
  real(kind=8) :: r8_normal_sample, r8_normal_01_sample
  real(kind=8), dimension(simlength) :: avgrew

  call date_and_time(time = timeval1)
  ! MAIN PROGRAM HERE
  write(*,*) "Start time"
  write(*,*) timeval1

  epsg = (/ real(0.0078125, kind=8), real(0.015625, kind=8), real(0.03125, kind=8), &
       real(0.0625, kind=8), real(0.125, kind=8), real(0.25, kind=8) /)
  cs = (/ real(0.0625, kind=8), real(0.125, kind=8), real(0.25, kind=8), real(0.5, kind=8), &
       real(1.0, kind=8), real(2.0, kind=8), real(4.0, kind=8) /)
  alphas = (/ real(0.03125, kind=8), real(0.0625, kind=8), real(0.125, kind=8), real(0.25, &
       kind=8), real(0.5, kind=8), real(1.0, kind=8), real(2.0, kind=8), real(4.0, kind=8) /)
  qs = (/ real(0.25, kind=8), real(0.5, kind=8), real(1.0, kind=8), real(2.0, kind=8), &
       real(4.0, kind=8) /)

  write(*,*) "Epsilon greedy started"

  ! epsilon greedy -- 4.633 seconds; 3.399 seconds
  do i=1, epslen
    do j=1, simlength
      q_stars = set_q()
      avgrew(j) = sum(karm_epsg(q_stars, steplen, epsg(i))) / real(steplen/2, kind=8)
    end do ! j
    avgReps(i) = sum(avgrew) / real(simlength, kind=8)
  end do

  call date_and_time(time = temptime)
  write(*,*) "Epsilon greedy done"
  write(*,*) temptime
  write(*,*) "Upper confidence bound started"

  ! upper confidence bound -- 8.921 seconds; 4.445
  do i=1, ucblen
    do j=1, simlength
      q_stars = set_q()
      avgrew(j) = sum(karm_ucb(q_stars, steplen, cs(i))) / real(steplen/2, kind=8)
    end do ! j
    avgRucb(i) = sum(avgrew) / real(simlength, kind=8)
  end do ! i

  call date_and_time(time = temptime)
  write(*,*) "Upper confidence bound done"
  write(*,*) temptime
  write(*,*) "Gradient bandit started"

  ! gradient bandit algorithm - 22.8 seconds; 6.804
 do i=1, gbalen
    do j=1, simlength
      q_stars = set_q()
      avgrew(j) = sum(karm_gba(q_stars, steplen, alphas(i))) / real(steplen/2, kind=8)
    end do ! j
    avgRgba(i) = sum(avgrew) / real(simlength, kind=8)
  end do ! i

  call date_and_time(time = temptime)
  write(*,*) "Gradient bandit done"
  write(*,*) temptime
  write(*,*) "Optimistic initial value started"

  ! optimistic initial value - 2.982; 2.797
  do i=1, oivlen
    do j=1, simlength
      q_stars = set_q()
      avgrew(j) = sum(karm_oiv(q_stars, steplen, qs(i))) / real(steplen/2, kind=8)
    end do ! j
    avgRoiv(i) = sum(avgrew) / real(simlength, kind=8)
  end do ! i

  call date_and_time(time = temptime)
  write(*,*) "Optimistic initial value done"
  write(*,*) temptime

  timeval2 = temptime
  write(*,*) "Total elapsed interval"
  write(*,*) timeval1
  write(*,*) timeval2

  ! save stuff to file
  open(unit = 8, file = "output_rl.txt", status = "new", action = "readwrite", iostat = ierror)

  ! write stuff to file
  write(unit = 8, fmt = *) avgReps(:)
  write(unit = 8, fmt = *) avgRucb(:)
  write(unit = 8, fmt = *) avgRgba(:)
  write(unit = 8, fmt = *) avgRoiv(:)
  
  ! close file down
  close(unit = 8)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! END MAIN PROGRAM
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! functions for use
  contains

!! EXTRA FUNCTIONS
  function set_q() result(q)
    implicit none
    real(kind=8), dimension(qlen) :: q
    integer :: ii
    real(kind=8) :: r8_normal_01_sample

    do ii=1, qlen
      q(ii) = real(0.0, kind=8)!r8_normal_01_sample()
    end do ! ii
  end function set_q

  function ignuin_jh ( low, high )
    implicit none
    integer ( kind = 4 ) err
    integer ( kind = 4 ) high
    integer ( kind = 4 ) i4_uni ! i4_uniform
    integer ( kind = 4 ) ign
    integer ( kind = 4 ) ignuin_jh
    integer ( kind = 4 ) low
    integer ( kind = 4 ) maxnow
    integer ( kind = 4 ) maxnum
    parameter ( maxnum = 2147483561 )
    integer ( kind = 4 ) ranp1
    integer ( kind = 4 ) width
    if ( high < low ) then
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'IGNUIN - Fatal error!'
      write ( *, '(a)' ) '  HIGH < LOW.'
      stop
    end if
    width = high - low
    if ( maxnum < width ) then
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'IGNUIN - Fatal error!'
      write ( *, '(a)' ) '  Range HIGH-LOW is too large.'
      stop
    end if
    if ( low == high ) then
      ignuin_jh = low
      return
    end if
    ranp1 = width + 1
    maxnow = ( maxnum / ranp1 ) * ranp1
    do
      ign = i4_uni ( ) - 1 ! i4_uniform
      if ( ign <= maxnow ) then
        exit
      end if
    end do
    ignuin_jh = low + mod ( ign, ranp1 )
    return
  end function ignuin_jh
!! END EXTRA FUNCTIONS

!! MAIN PROGRAM SIMULATION FUNCTIONS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  function karm_ucb(q_s, numsteps, c) !result(rewards)
    implicit none

    real(kind=8), dimension(qlen) :: q_s
    integer :: numsteps
    real(kind=8) :: c
    real(kind=8), dimension(numsteps/2) :: karm_ucb

    real(kind=8), dimension(qlen) :: Q, n, Qplus
    real(kind=8), dimension(numsteps) :: rewards
    real(kind=8) :: R
    integer :: A, ii, jj

    real(kind=8) :: r8_normal_sample

    ! initialize Q and n
    do ii=1, qlen
      Q(ii) = real(0.0, kind=8)
      n(ii) = real(0.0, kind=8)
    end do

    ! first 10
    do ii=1, qlen
      R = r8_normal_sample(q_s(ii), real(1.0, kind=8))
      n(ii) = n(ii) + real(1.0, kind=8)
      Q(ii) = Q(ii) + real(1.0, kind=8) / n(ii) * (R - Q(ii))
      rewards(ii) = R
    end do
    ! 11 and on
    do ii = qlen+1, numsteps
      do jj = 1, qlen
        Qplus(jj) = Q(jj) + c * dsqrt(dlog(real(ii, kind=8)) / n(jj))
      end do
      A = maxloc(Qplus, 1)
      R = r8_normal_sample(q_s(A), real(1.0, kind=8))
      n(A) = n(A) + real(1.0, kind=8)
      Q(A) = Q(A) + (real(1.0, kind=8) / n(A)) * (R - Q(A))
      rewards(ii) = R
      ! add noise to q_s
      do jj = 1, qlen
        q_s(jj) = q_s(jj) + r8_normal_sample(real(0.0, kind=8), real(0.01, kind=8))
      end do ! jj
    end do ! ii
    karm_ucb = rewards((numsteps/2 + 1):numsteps)
  end function karm_ucb

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  function karm_epsg(q_s, numsteps, eps) !result(rewards)
    implicit none

    real(kind=8), dimension(qlen) :: q_s
    integer :: numsteps
    real(kind=8) :: eps
    real(kind=8), dimension(numsteps/2) :: karm_epsg

    real(kind=8), dimension(qlen) :: Q, n
    real(kind=8), dimension(numsteps) :: rewards
    real(kind=8) :: R, prob
    integer :: A, ii, jj, maxnum, tempA, tempn
    integer, dimension(qlen) :: maxes

    real(kind=8) :: r8_normal_sample, r8_uniform_01_sample

    ! initialize Q and n
    do ii=1, qlen
      Q(ii) = real(0.0, kind=8)
      n(ii) = real(0.0, kind=8)
    end do ! ii

    ! main do loop
    do ii=1, numsteps
      prob = r8_uniform_01_sample()
      if (prob < eps) then
        A = ignuin_jh(1, qlen)
      else if (maxval(Q) == 0.0) then
        ! first find out how many maxes there are and where
        maxnum = 0
        do jj=1, qlen
          maxes(jj) = 0
          if (Q(jj) == maxval(Q)) then
            maxes(jj) = 1
            maxnum = maxnum + 1
          end if
        end do ! jj
        ! select which max to pick
        tempA = ignuin_jh(1, maxnum)
        ! find that max
        maxnum = 0
        tempn = 1
        do while (maxnum < tempA)
          if (maxes(tempn) == 1) then
            maxnum = maxnum + 1
            tempn = tempn + 1
          else
            tempn = tempn + 1
          end if
        end do ! while
        A = tempn - 1
      else 
        A = maxloc(Q, 1)
      end if
      R = r8_normal_sample(q_s(A), real(1.0, kind=8))
      n(A) = n(A) + real(1.0, kind=8)
      Q(A) = Q(A) + (real(1.0, kind=8) / n(A)) * (R - Q(A))
      rewards(ii) = R
      ! add noise to q_s
      do jj = 1, qlen
        q_s(jj) = q_s(jj) + r8_normal_sample(real(0.0, kind=8), real(0.01, kind=8))
      end do ! jj
    end do ! ii
    karm_epsg = rewards((numsteps/2 + 1):numsteps)
  end function karm_epsg

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  function karm_oiv(q_s, numsteps, Q0) !result(rewards)
    implicit none

    real(kind=8), dimension(qlen) :: q_s
    integer :: numsteps
    real(kind=8) :: Q0
    real(kind=8), dimension(numsteps/2) :: karm_oiv

    real(kind=8), dimension(qlen) :: Q
    real(kind=8), dimension(numsteps) :: rewards
    real(kind=8) :: R
    integer :: A, ii, jj, maxnum, tempA, tempn
    integer, dimension(qlen) :: maxes

    real(kind=8) :: r8_normal_sample

    ! initialize Q
    do ii=1, qlen
      Q(ii) = Q0
    end do ! ii

    ! main do loop
    do ii=1, numsteps
      if (maxval(Q) == Q0) then
        ! first find out how many maxes there are and where
        maxnum = 0
        do jj=1, qlen
          maxes(jj) = 0
          if (Q(jj) == maxval(Q)) then
            maxes(jj) = 1
            maxnum = maxnum + 1
          end if
        end do ! jj
        ! select which max to pick
        tempA = ignuin_jh(1, maxnum)
        ! find that max
        maxnum = 0
        tempn = 1
        do while (maxnum < tempA)
          if (maxes(tempn) == 1) then
            maxnum = maxnum + 1
            tempn = tempn + 1
          else
            tempn = tempn + 1
          end if
        end do ! while
        ! set A to that max
        A = tempn - 1
      else ! if the max is no longer Q0, find the max
        A = maxloc(Q, 1)
      end if
      R = r8_normal_sample(q_s(A), real(1.0, kind=8))
      Q(A) = Q(A) + 0.1 * (R - Q(A)) ! alpha = 0.1 here
      rewards(ii) = R
      ! add noise to q_s
      do jj = 1, qlen
        q_s(jj) = q_s(jj) + r8_normal_sample(real(0.0, kind=8), real(0.01, kind=8))
      end do ! jj
    end do ! ii
    karm_oiv = rewards((numsteps/2 + 1):numsteps)
  end function karm_oiv

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  function karm_gba(q_s, numsteps, alpha) !result(rewards)
    implicit none

    real(kind=8), dimension(qlen) :: q_s
    integer :: numsteps
    real(kind=8) :: alpha
    real(kind=8), dimension(numsteps/2) :: karm_gba

    real(kind=8), dimension(qlen) :: H
    real(kind=4), dimension(qlen) :: probs
    real(kind=8), dimension(numsteps) :: rewards
    real(kind=8) :: R, Rbar, Hbar
    real(kind=4) :: sumexpH, sumprobs
    integer :: A, ii, jj, maxnum, tempn
    integer(kind=4), dimension(qlen) :: maxes

    real(kind=8) :: r8_normal_sample

    ! initialize H, Rbar, probs
    do ii=1, qlen
      H(ii) = real(0.0, kind=8)
      probs(ii) = real(1.0, kind=4) / real(qlen, kind=4)
    end do ! ii
    Rbar = real(0.0, kind=8)
    !main do loop
    do ii=1, numsteps
      ! select which max to pick
      do jj = 1, qlen
        maxes(jj) = 0
      end do
      call genmul(1, probs(1:(qlen-1)), qlen, maxes) !n, p, ncat, ix; -- _jh
      A = maxloc(maxes, 1)
      R = r8_normal_sample(q_s(A), real(1.0, kind=8))
      Rbar = Rbar + (real(1.0, kind=8) / real(ii, kind=8)) * (R - Rbar)
      ! update H and probs
      do jj=1, qlen
        if (.NOT. (jj == A)) then
          H(jj) = H(jj) - alpha * (R - Rbar) * real(probs(jj), kind=8)
        end if
      end do
      H(A) = H(A) + alpha * (R - Rbar) * (1.0 - real(probs(A), kind=8))
      sumexpH = real(0.0, kind=4)
      do jj = 1, qlen
        sumexpH = sumexpH + exp(real(H(jj), kind=4))
      end do
      ! update probs
      do jj = 1, qlen
        probs(jj) = exp(real(H(jj), kind=4)) / sumexpH
      end do
      ! bring probs into range
      do jj = 1, qlen
        ! bound away from zero and one - IGNBIN
        probs(jj) = amin1(amax1(probs(jj), real(1E-6, kind=4)), real(0.99999, kind=4))
      end do ! jj
      sumprobs = sum(probs)
      ! renormalize to 0.9999 - GENMUL
      do jj = 1, qlen
        probs(jj) = probs(jj) / sum(probs) * real(0.9999, kind=4)
      end do
      rewards(ii) = R
      ! add noise to q_s
      do jj = 1, qlen
        q_s(jj) = q_s(jj) + r8_normal_sample(real(0.0, kind=8), real(0.01, kind=8))
      end do ! jj
    end do ! ii
    karm_gba = rewards((numsteps/2 + 1):numsteps)
  end function karm_gba

end program reinforce
